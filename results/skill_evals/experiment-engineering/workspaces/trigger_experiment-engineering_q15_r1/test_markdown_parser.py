import unittest
from markdown_parser import parse_headings


class TestParseHeadings(unittest.TestCase):
    """Unit tests for the markdown parser module."""

    def test_single_heading_level_1(self):
        """Test parsing a single level 1 heading."""
        text = "# Main Title"
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["level"], 1)
        self.assertEqual(result[0]["title"], "Main Title")

    def test_all_heading_levels(self):
        """Test parsing headings at all valid levels (1-6)."""
        text = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6"""
        result = parse_headings(text)
        self.assertEqual(len(result), 6)
        for i, expected_level in enumerate(range(1, 7)):
            self.assertEqual(result[i]["level"], expected_level)
            self.assertEqual(result[i]["title"], f"Level {expected_level}")

    def test_empty_input(self):
        """Test parsing empty string returns empty list."""
        result = parse_headings("")
        self.assertEqual(result, [])

    def test_no_headings(self):
        """Test text with no headings returns empty list."""
        text = """This is just regular text.
Some more text here.
No headings at all."""
        result = parse_headings(text)
        self.assertEqual(result, [])

    def test_mixed_content(self):
        """Test parsing text with both headings and non-heading content."""
        text = """# Introduction
Some introductory text here.
More regular text.

## Section 1
Content under section 1.

### Subsection 1.1
More content.

## Section 2
Final content."""
        result = parse_headings(text)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["title"], "Introduction")
        self.assertEqual(result[1]["title"], "Section 1")
        self.assertEqual(result[2]["title"], "Subsection 1.1")
        self.assertEqual(result[3]["title"], "Section 2")

    def test_heading_with_trailing_spaces(self):
        """Test that trailing spaces in headings are stripped."""
        text = "# Title with trailing spaces   "
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], "Title with trailing spaces")

    def test_heading_with_leading_spaces_after_hash(self):
        """Test heading with multiple spaces between hash and title."""
        text = "#    Title with spaces after hash"
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], "Title with spaces after hash")

    def test_heading_with_special_characters(self):
        """Test headings containing special characters."""
        text = "# Title with @#$%^&*() special chars!"
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], "Title with @#$%^&*() special chars!")

    def test_heading_with_inline_code(self):
        """Test heading containing inline code."""
        text = "# Install `package_name` from PyPI"
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], "Install `package_name` from PyPI")

    def test_heading_with_markdown_links(self):
        """Test heading containing markdown link syntax."""
        text = "# Read [Documentation](https://example.com)"
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], "Read [Documentation](https://example.com)")

    def test_invalid_heading_no_space_after_hash(self):
        """Test that heading without space after hash is not parsed."""
        text = "#NoSpace"
        result = parse_headings(text)
        self.assertEqual(result, [])

    def test_invalid_heading_too_many_hashes(self):
        """Test that 7+ hashes do not form valid heading."""
        text = "####### Invalid Level"
        result = parse_headings(text)
        self.assertEqual(result, [])

    def test_invalid_heading_zero_hashes(self):
        """Test that text starting with space and hash is not a heading."""
        text = " # Not a heading"
        result = parse_headings(text)
        self.assertEqual(result, [])

    def test_empty_heading(self):
        """Test parsing heading with only hash and no title."""
        text = "# "
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["level"], 1)
        self.assertEqual(result[0]["title"], "")

    def test_heading_with_only_spaces_as_title(self):
        """Test heading with only spaces as title gets stripped to empty."""
        text = "#    "
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "")

    def test_multiple_headings_same_level(self):
        """Test multiple headings at the same level."""
        text = """# First
# Second
# Third"""
        result = parse_headings(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["title"], "First")
        self.assertEqual(result[1]["title"], "Second")
        self.assertEqual(result[2]["title"], "Third")

    def test_deeply_nested_structure(self):
        """Test deeply nested heading structure (all levels mixed)."""
        text = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6
## Back to Level 2
# Back to Level 1"""
        result = parse_headings(text)
        self.assertEqual(len(result), 8)
        expected_levels = [1, 2, 3, 4, 5, 6, 2, 1]
        for i, expected in enumerate(expected_levels):
            self.assertEqual(result[i]["level"], expected)

    def test_heading_with_numbers(self):
        """Test heading containing numbers."""
        text = "# Chapter 42: The Answer"
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], "Chapter 42: The Answer")

    def test_heading_with_punctuation(self):
        """Test heading with various punctuation marks."""
        text = "# Title: Subtitle - Part 1 (Extended)"
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], "Title: Subtitle - Part 1 (Extended)")

    def test_windows_line_endings(self):
        """Test parsing text with Windows line endings (CRLF)."""
        text = "# Heading 1\r\nSome text\r\n## Heading 2"
        result = parse_headings(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Heading 1")
        self.assertEqual(result[1]["title"], "Heading 2")

    def test_mac_line_endings(self):
        """Test parsing text with old Mac line endings (CR only)."""
        text = "# Heading 1\rSome text\r## Heading 2"
        result = parse_headings(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Heading 1")
        self.assertEqual(result[1]["title"], "Heading 2")

    def test_unicode_characters_in_heading(self):
        """Test heading with unicode characters."""
        text = "# 你好世界 - Hello World - 🌍"
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], "你好世界 - Hello World - 🌍")

    def test_heading_with_tabs(self):
        """Test heading with tab character after hash."""
        text = "#\tTabbed Title"
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Tabbed Title")

    def test_consecutive_hashes_in_title(self):
        """Test heading with hash characters in the title itself."""
        text = "## Section #1: #hashtag"
        result = parse_headings(text)
        self.assertEqual(result[0]["level"], 2)
        self.assertEqual(result[0]["title"], "Section #1: #hashtag")

    def test_html_comment_not_heading(self):
        """Test that HTML comments are not parsed as headings."""
        text = "<!-- # This is a comment -->"
        result = parse_headings(text)
        self.assertEqual(result, [])

    def test_code_block_not_parsed(self):
        """Test that headings in code blocks are still captured (regex doesn't know context)."""
        text = """```markdown
# Heading in code block
```"""
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Heading in code block")

    def test_very_long_heading(self):
        """Test parsing very long heading title."""
        long_title = "A" * 1000
        text = f"# {long_title}"
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], long_title)

    def test_return_type_is_list(self):
        """Test that return value is always a list."""
        self.assertIsInstance(parse_headings(""), list)
        self.assertIsInstance(parse_headings("# Test"), list)

    def test_return_items_are_dicts(self):
        """Test that each item in result is a dictionary."""
        text = "# Test"
        result = parse_headings(text)
        self.assertIsInstance(result[0], dict)

    def test_return_dict_has_required_keys(self):
        """Test that returned dictionaries have 'level' and 'title' keys."""
        text = "# Test"
        result = parse_headings(text)
        self.assertIn("level", result[0])
        self.assertIn("title", result[0])
        self.assertEqual(len(result[0]), 2)

    def test_level_is_integer(self):
        """Test that level value is an integer."""
        text = "## Test"
        result = parse_headings(text)
        self.assertIsInstance(result[0]["level"], int)

    def test_title_is_string(self):
        """Test that title value is a string."""
        text = "# Test"
        result = parse_headings(text)
        self.assertIsInstance(result[0]["title"], str)


if __name__ == "__main__":
    unittest.main()
