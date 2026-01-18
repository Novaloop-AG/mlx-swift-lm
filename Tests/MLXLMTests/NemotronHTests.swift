// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

public class NemotronHTests: XCTestCase {

    /// Create a minimal test configuration for NemotronH
    /// Uses small dimensions to keep tests fast
    private func makeTestConfig(pattern: String = "M*M-E") -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 100,
            hiddenSize: 64,
            numHiddenLayers: pattern.count,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: pattern,
            layerNormEpsilon: 1e-5,
            nGroup: 2,
            topkGroup: 1
        )
    }

    // MARK: - Basic Forward Pass Tests

    func testNemotronHForwardPass() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    func testNemotronHWithMambaOnly() throws {
        let config = makeTestConfig(pattern: "MMM")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHWithAttentionOnly() throws {
        let config = makeTestConfig(pattern: "***")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHWithMLP() throws {
        let config = makeTestConfig(pattern: "M-*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHWithMoE() throws {
        let config = makeTestConfig(pattern: "ME*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    func testNemotronHFullPattern() throws {
        // Test a pattern with all block types
        let config = makeTestConfig(pattern: "M-E*M-E*")
        let model = NemotronHModel(config)

        let input = MLXArray([1, 2, 3, 4])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 4, 100])
    }

    // MARK: - Cache Tests

    func testNemotronHCacheCreation() throws {
        // Pattern: M*M- has 2 Mamba + 1 Attention = 3 caches
        let config = makeTestConfig(pattern: "M*M-")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // Only Mamba (M) and Attention (*) layers have caches
        // Pattern M*M- has M, *, M = 3 cacheable layers
        XCTAssertEqual(cache.count, 3)
    }

    func testNemotronHCacheCountMambaOnly() throws {
        let config = makeTestConfig(pattern: "MMM")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // 3 Mamba layers = 3 caches
        XCTAssertEqual(cache.count, 3)
    }

    func testNemotronHCacheCountAttentionOnly() throws {
        let config = makeTestConfig(pattern: "***")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // 3 Attention layers = 3 caches
        XCTAssertEqual(cache.count, 3)
    }

    func testNemotronHCacheCountMixed() throws {
        // Pattern with MLP (-) and MoE (E) which don't have caches
        let config = makeTestConfig(pattern: "M-E*-E")
        let model = NemotronHModel(config)

        let cache = model.newCache(parameters: nil)

        // Only M and * have caches: M, * = 2 caches
        XCTAssertEqual(cache.count, 2)
    }

    // MARK: - Incremental Generation Tests

    func testNemotronHIncrementalGeneration() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // First pass - process prompt
        let prompt = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let cache = model.newCache(parameters: nil)
        let promptOutput = model.callAsFunction(prompt, cache: cache)

        XCTAssertEqual(promptOutput.shape, [1, 5, 100])

        // Second pass - generate next token
        let nextToken = MLXArray([6])[.newAxis, .ellipsis]
        let nextOutput = model.callAsFunction(nextToken, cache: cache)

        XCTAssertEqual(nextOutput.shape, [1, 1, 100])
    }

    // MARK: - KV Heads Tests

    func testNemotronHKVHeads() throws {
        let config = makeTestConfig(pattern: "M*M*")
        let model = NemotronHModel(config)

        // kvHeads should have entries for Mamba (0) and Attention (numKeyValueHeads)
        // Pattern M*M* = [0, 2, 0, 2] where 2 is numKeyValueHeads
        XCTAssertEqual(model.kvHeads.count, 4)
        XCTAssertEqual(model.kvHeads[0], 0)  // Mamba
        XCTAssertEqual(model.kvHeads[1], 2)  // Attention
        XCTAssertEqual(model.kvHeads[2], 0)  // Mamba
        XCTAssertEqual(model.kvHeads[3], 2)  // Attention
    }

    // MARK: - Vocabulary Size Tests

    func testNemotronHVocabularySize() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        XCTAssertEqual(model.vocabularySize, 100)
    }

    // MARK: - Batch Processing Tests

    func testNemotronHBatchProcessing() throws {
        let config = makeTestConfig(pattern: "M*")
        let model = NemotronHModel(config)

        // Batch of 2 sequences - use reshaped to create 2D input
        let flat = MLXArray([1, 2, 3, 4, 5, 6])
        let input = flat.reshaped(2, 3)
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [2, 3, 100])
    }
}
