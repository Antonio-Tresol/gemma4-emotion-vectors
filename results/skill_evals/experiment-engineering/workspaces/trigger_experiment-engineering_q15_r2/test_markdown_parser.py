import unittest
from markdown_parser import parse_headings, HEADING


class TestHeadingRegex(unittest.TestCase):
    """Test the HEADING regex pattern."""

    def test_h1_heading(self):
        """Test matching H1 heading."""
        match = HEADING.match("# Hello World")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "#")
        self.assertEqual(match.group(2), "Hello World")

    def test_h6_heading(self):
        """Test matching H6 heading."""
        match = HEADING.match("###### Small heading")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "######")
        self.assertEqual(match.group(2), "Small heading")

    def test_all_heading_levels(self):
        """Test matching headings at all levels."""
        for level in range(1, 7):
            hashes = "#" * level
            match = HEADING.match(f"{hashes} Level {level} heading")
            self.assertIsNotNone(match)
            self.assertEqual(len(match.group(1)), level)

    def test_heading_with_extra_spaces(self):
        """Test heading with multiple spaces after hashes."""
        match = HEADING.match("#  Multiple spaces")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(2), "Multiple spaces")

    def test_heading_with_special_characters(self):
        """Test heading with special characters."""
        match = HEADING.match("## Title with @#$% special chars!")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(2), "Title with @#$% special chars!")

    def test_no_space_after_hashes_no_match(self):
        """Test that heading without space after hashes doesn't match."""
        match = HEADING.match("#NoSpace")
        self.assertIsNone(match)

    def test_too_many_hashes_no_match(self):
        """Test that more than 6 hashes don't match."""
        match = HEADING.match("####### Too many")
        self.assertIsNone(match)

    def test_no_hash_no_match(self):
        """Test that line without hash doesn't match."""
        match = HEADING.match("Regular text")
        self.assertIsNone(match)

    def test_hash_in_middle_no_match(self):
        """Test that hash not at start doesn't match."""
        match = HEADING.match("Some text # Not a heading")
        self.assertIsNone(match)

    def test_empty_string_no_match(self):
        """Test that empty string doesn't match."""
        match = HEADING.match("")
        self.assertIsNone(match)

    def test_heading_with_trailing_whitespace(self):
        """Test heading with trailing whitespace."""
        match = HEADING.match("## Title   ")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(2), "Title   ")


class TestParseHeadings(unittest.TestCase):
    """Test the parse_headings function."""

    def test_single_h1_heading(self):
        """Test parsing a single H1 heading."""
        text = "# Main Title"
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["level"], 1)
        self.assertEqual(result[0]["title"], "Main Title")

    def test_multiple_headings_different_levels(self):
        """Test parsing multiple headings at different levels."""
        text = """# Chapter 1
## Section 1.1
### Subsection 1.1.1
## Section 1.2"""
        result = parse_headings(text)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["level"], 1)
        self.assertEqual(result[1]["level"], 2)
        self.assertEqual(result[2]["level"], 3)
        self.assertEqual(result[3]["level"], 2)

    def test_headings_with_regular_text(self):
        """Test parsing text with both headings and regular content."""
        text = """# Title
This is regular text.
## Subtitle
More text here.
### Another heading
Even more text."""
        result = parse_headings(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["title"], "Title")
        self.assertEqual(result[1]["title"], "Subtitle")
        self.assertEqual(result[2]["title"], "Another heading")

    def test_empty_string(self):
        """Test parsing empty string."""
        result = parse_headings("")
        self.assertEqual(result, [])

    def test_text_with_no_headings(self):
        """Test parsing text with no headings."""
        text = """Just regular text.
More text without headings.
Even more text."""
        result = parse_headings(text)
        self.assertEqual(result, [])

    def test_heading_with_special_characters(self):
        """Test parsing heading with special characters."""
        text = "## Title with @#$% special chars!"
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Title with @#$% special chars!")

    def test_heading_title_stripped(self):
        """Test that heading title is stripped of whitespace."""
        text = "##   Title with spaces   "
        result = parse_headings(text)
        self.assertEqual(result[0]["title"], "Title with spaces")

    def test_all_heading_levels_parsed(self):
        """Test parsing all heading levels."""
        text = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6"""
        result = parse_headings(text)
        self.assertEqual(len(result), 6)
        for i, heading in enumerate(result, 1):
            self.assertEqual(heading["level"], i)
            self.assertEqual(heading["title"], f"Level {i}")

    def test_consecutive_headings(self):
        """Test parsing consecutive headings without text between."""
        text = """# Heading 1
## Heading 2
### Heading 3"""
        result = parse_headings(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["title"], "Heading 1")
        self.assertEqual(result[1]["title"], "Heading 2")
        self.assertEqual(result[2]["title"], "Heading 3")

    def test_heading_with_code_like_syntax(self):
        """Test heading containing code-like syntax."""
        text = "## Function `my_func()` documentation"
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Function `my_func()` documentation")

    def test_heading_with_markdown_links(self):
        """Test heading containing markdown link syntax."""
        text = "## See [documentation](https://example.com)"
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "See [documentation](https://example.com)")

    def test_text_with_hash_not_heading(self):
        """Test that hash in the middle of line is not parsed as heading."""
        text = """Some text with # hash in the middle
## Real heading
#NoSpace is not a heading"""
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Real heading")

    def test_multiline_text_with_windows_line_endings(self):
        """Test parsing text with Windows line endings."""
        text = "# Heading 1\r\n## Heading 2\r\nRegular text\r\n### Heading 3"
        result = parse_headings(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["level"], 1)
        self.assertEqual(result[1]["level"], 2)
        self.assertEqual(result[2]["level"], 3)

    def test_unicode_in_headings(self):
        """Test parsing headings with unicode characters."""
        text = """# Café
## Résumé français
### 日本語タイトル
#### Emoji 🚀 Test"""
        result = parse_headings(text)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["title"], "Café")
        self.assertEqual(result[1]["title"], "Résumé français")
        self.assertEqual(result[2]["title"], "日本語タイトル")
        self.assertEqual(result[3]["title"], "Emoji 🚀 Test")

    def test_very_long_heading_title(self):
        """Test parsing heading with very long title."""
        long_title = "A" * 1000
        text = f"# {long_title}"
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], long_title)

    def test_large_document_performance(self):
        """Test parsing large document with many headings."""
        text_lines = []
        expected_count = 0
        for i in range(1, 101):
            text_lines.append(f"# Heading {i}")
            expected_count += 1
            text_lines.append(f"Some regular text after heading {i}.")

        text = "\n".join(text_lines)
        result = parse_headings(text)
        self.assertEqual(len(result), expected_count)

    def test_single_newline(self):
        """Test parsing single newline character."""
        result = parse_headings("\n")
        self.assertEqual(result, [])

    def test_multiple_newlines(self):
        """Test parsing multiple consecutive newlines."""
        text = "# Heading 1\n\n\n## Heading 2"
        result = parse_headings(text)
        self.assertEqual(len(result), 2)

    def test_heading_at_end_of_document(self):
        """Test heading as last line of document."""
        text = """Some text
More text
# Final Heading"""
        result = parse_headings(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Final Heading")

    def test_dict_structure(self):
        """Test that returned dictionaries have correct structure."""
        text = "## Test Heading"
        result = parse_headings(text)
        self.assertEqual(set(result[0].keys()), {"level", "title"})
        self.assertIsInstance(result[0]["level"], int)
        self.assertIsInstance(result[0]["title"], str)


if __name__ == "__main__":
    unittest.main()
