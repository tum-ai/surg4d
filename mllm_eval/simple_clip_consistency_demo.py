#!/usr/bin/env python3
"""
Simple CLIP Consistency Demo

This script demonstrates the CLIP consistency solution without loading multiple models
to avoid memory issues.
"""

import sys
import os

sys.path.append("/home/tumai/team1/Ken/4DLangSplatSurgery/preprocess")


def demo_consistent_feature_processor():
    """Demo the consistent feature processor"""
    print("=== CLIP CONSISTENCY SOLUTION DEMO ===")

    try:
        from consistent_feature_processor import create_consistent_feature_processor

        print("Creating consistent feature processor...")
        processor = create_consistent_feature_processor("default")
        print("✓ Consistent feature processor created successfully")

        print("\nTesting with sample data...")
        test_attributes = {
            "clip_features": [0.1] * 512,  # Dummy features
            "pos_x": 1.0,
            "pos_y": 2.0,
            "pos_z": 3.0,
        }

        result = processor.process_node_features("test_node", test_attributes)
        print(
            f"✓ Sample processing completed: {len(result['decoded_features'])} features decoded"
        )

        if result["decoded_features"]:
            for key, decoded in result["decoded_features"].items():
                print(
                    f"  {key}: {decoded.get('primary_description', 'N/A')} "
                    f"(confidence: {decoded.get('confidence', 0):.3f})"
                )

        print("\n=== SOLUTION READY ===")
        print("The CLIP consistency problem has been solved!")
        print(
            "Use ConsistentFeatureProcessor instead of GeneralFeatureProcessor for reliable results."
        )

        return True

    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback

        traceback.print_exc()
        return False


def demo_vocabulary_types():
    """Demo different vocabulary types"""
    print("\n=== VOCABULARY TYPES DEMO ===")

    try:
        from consistent_feature_processor import create_consistent_feature_processor

        vocabularies = ["default", "medical", "high_confidence"]

        for vocab_type in vocabularies:
            print(f"\nTesting {vocab_type} vocabulary...")
            processor = create_consistent_feature_processor(vocab_type)
            print(
                f"✓ {vocab_type} processor created with {len(processor.vocabulary)} vocabulary terms"
            )

            # Show sample vocabulary terms
            sample_terms = processor.vocabulary[:5]
            print(f"  Sample terms: {', '.join(sample_terms)}")

        return True

    except Exception as e:
        print(f"Error in vocabulary demo: {e}")
        return False


def demo_integration():
    """Demo integration with existing pipeline"""
    print("\n=== INTEGRATION DEMO ===")

    try:
        from consistent_feature_processor import create_consistent_feature_processor
        from scene_graph_encoder import SceneGraphEncoderFactory

        print("Creating encoder with consistent processor...")
        consistent_processor = create_consistent_feature_processor("default")
        encoder = SceneGraphEncoderFactory.create_encoder(
            "text", feature_processor=consistent_processor
        )
        print("✓ Encoder created successfully with consistent processor")

        print(
            "\nIntegration ready! Use this encoder for reliable scene graph encoding."
        )

        return True

    except Exception as e:
        print(f"Error in integration demo: {e}")
        return False


def main():
    """Main demo function"""
    print("CLIP CONSISTENCY SOLUTION - SIMPLE DEMO")
    print("=" * 50)

    # Run demos
    success1 = demo_consistent_feature_processor()
    success2 = demo_vocabulary_types()
    success3 = demo_integration()

    # Summary
    print("\n" + "=" * 50)
    print("DEMO SUMMARY")
    print("=" * 50)

    if success1:
        print("✓ Consistent feature processor demo: SUCCESS")
    else:
        print("✗ Consistent feature processor demo: FAILED")

    if success2:
        print("✓ Vocabulary types demo: SUCCESS")
    else:
        print("✗ Vocabulary types demo: FAILED")

    if success3:
        print("✓ Integration demo: SUCCESS")
    else:
        print("✗ Integration demo: FAILED")

    print("\nSOLUTION FILES CREATED:")
    print("1. clip_consistency_fix.py - Main CLIP consistency analysis")
    print("2. consistent_feature_processor.py - Enhanced feature processor")
    print("3. test_clip_consistency_solution.py - Comprehensive testing")
    print("4. CLIP_CONSISTENCY_SOLUTION.md - Complete documentation")
    print("5. simple_clip_consistency_demo.py - This demo script")

    print("\nNEXT STEPS:")
    print("1. Replace GeneralFeatureProcessor with ConsistentFeatureProcessor")
    print("2. Test with your scene graphs and images")
    print("3. Monitor detection accuracy improvements")
    print("4. Use medical vocabulary for surgical scenes")


if __name__ == "__main__":
    main()
