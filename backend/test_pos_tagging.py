"""Test script for POS Tagging implementation."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all POS tagging components can be imported."""
    print("Testing POS Tagging imports...")

    try:
        from algorithms.nlp.pos_tagging import (
            POSTaggingModel,
            POSTaggingParameters,
            POSTaggingResponse,
            TaggedWord,
            DependencyEdge,
            get_sample_sentences,
            get_pos_tag_descriptions,
            get_pos_tag_colors,
            get_dataset_info
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_functions():
    """Test data loading functions."""
    print("\nTesting data functions...")

    try:
        from algorithms.nlp.pos_tagging import (
            get_sample_sentences,
            get_pos_tag_descriptions,
            get_pos_tag_colors,
            get_dataset_info
        )

        # Test sample sentences
        sentences = get_sample_sentences()
        assert len(sentences) >= 25, f"Expected at least 25 sentences, got {len(sentences)}"
        assert all('id' in s and 'text' in s for s in sentences), "All sentences should have 'id' and 'text'"
        print(f"✓ Sample sentences: {len(sentences)} sentences loaded")

        # Test tag descriptions
        descriptions = get_pos_tag_descriptions()
        assert 'NOUN' in descriptions, "Universal tags should be present"
        assert 'NN' in descriptions, "Penn Treebank tags should be present"
        print(f"✓ Tag descriptions: {len(descriptions)} tags described")

        # Test tag colors
        colors = get_pos_tag_colors()
        assert 'NOUN' in colors, "Colors should include universal tags"
        assert all(c.startswith('#') for c in colors.values()), "All colors should be hex codes"
        print(f"✓ Tag colors: {len(colors)} color mappings")

        # Test dataset info
        info = get_dataset_info()
        assert 'name' in info and 'size' in info, "Dataset info should have name and size"
        print(f"✓ Dataset info: {info['name']}")

        return True

    except Exception as e:
        print(f"✗ Error in data functions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schema():
    """Test schema validation."""
    print("\nTesting schema...")

    try:
        from algorithms.nlp.pos_tagging import POSTaggingParameters, TaggedWord

        # Test valid parameters
        params = POSTaggingParameters(
            tagger='spacy',
            text_index=0,
            show_fine_grained=True,
            show_dependencies=True,
            tag_scheme='penn'
        )
        print(f"✓ Valid parameters created: {params.tagger}, text_index={params.text_index}")

        # Test with custom text
        params_custom = POSTaggingParameters(
            use_custom_text=True,
            custom_text="The cat sits on the mat."
        )
        print(f"✓ Custom text parameters created")

        # Test TaggedWord
        word = TaggedWord(
            text="cat",
            pos_coarse="NOUN",
            pos_fine="NN",
            tag="NN",
            description="Noun, singular",
            index=0,
            lemma="cat",
            is_stop=False
        )
        print(f"✓ TaggedWord created: {word.text} ({word.tag})")

        return True

    except Exception as e:
        print(f"✗ Error in schema: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test POS tagging model."""
    print("\nTesting POS tagging model...")

    try:
        from algorithms.nlp.pos_tagging import POSTaggingModel, POSTaggingParameters

        # Create model
        model = POSTaggingModel()
        print("✓ Model instantiated")

        # Test with sample text
        params = POSTaggingParameters(
            tagger='spacy',
            text_index=0,
            show_fine_grained=True,
            show_dependencies=True
        )

        result = model.process(params)

        if result.success:
            print(f"✓ POS tagging succeeded")
            print(f"  - Words tagged: {result.metrics.get('total_words', 0)}")
            print(f"  - Unique tags: {result.metrics.get('unique_pos_tags', 0)}")
            print(f"  - Execution time: {result.execution_time_ms:.2f}ms")
            print(f"  - Text: {result.text_analyzed[:50]}...")

            # Show sample tagged words
            if result.tagged_words:
                print("\n  Sample tagged words:")
                for word in result.tagged_words[:5]:
                    print(f"    {word.text:15} -> {word.tag:8} ({word.pos_coarse})")

            return True
        else:
            print(f"✗ POS tagging failed: {result.error}")
            return False

    except Exception as e:
        print(f"✗ Error in model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_registration():
    """Test that the algorithm is registered in the API."""
    print("\nTesting API registration...")

    try:
        from utils.algorithm_metadata import AlgorithmRegistry

        metadata = AlgorithmRegistry.get("pos-tagging")

        if metadata:
            print(f"✓ Algorithm registered: {metadata.name}")
            print(f"  - Category: {metadata.category}")
            print(f"  - Difficulty: {metadata.difficulty}")
            print(f"  - Parameters: {len(metadata.parameters)}")
            return True
        else:
            print("✗ Algorithm not registered")
            return False

    except Exception as e:
        print(f"✗ Error checking registration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("POS Tagging Implementation Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Data Functions", test_data_functions()))
    results.append(("Schema", test_schema()))
    results.append(("Model", test_model()))
    results.append(("API Registration", test_api_registration()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)
