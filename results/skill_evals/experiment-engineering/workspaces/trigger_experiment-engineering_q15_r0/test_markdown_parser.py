import pytest
from markdown_parser import parse_headings


class TestParseHeadings:
    """Unit tests for the markdown parser module."""

    def test_empty_text(self):
        """Test parsing empty text returns empty list."""
        assert parse_headings("") == []

    def test_text_with_no_headings(self):
        """Test parsing text without headings returns empty list."""
        text = "This is some regular text.\nWith multiple lines.\nBut no headings."
        assert parse_headings(text) == []

    def test_single_h1_heading(self):
        """Test parsing a single H1 heading."""
        text = "# Main Title"
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["level"] == 1
        assert result[0]["title"] == "Main Title"

    def test_single_h2_heading(self):
        """Test parsing a single H2 heading."""
        text = "## Subheading"
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["level"] == 2
        assert result[0]["title"] == "Subheading"

    def test_all_heading_levels(self):
        """Test parsing all valid heading levels (H1-H6)."""
        text = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6"""
        result = parse_headings(text)
        assert len(result) == 6
        for i, heading in enumerate(result, start=1):
            assert heading["level"] == i
            assert heading["title"] == f"Level {i}"

    def test_headings_with_extra_spaces(self):
        """Test parsing headings with multiple spaces before title."""
        text = "#   Title with spaces"
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["level"] == 1
        assert result[0]["title"] == "Title with spaces"

    def test_headings_with_trailing_spaces(self):
        """Test parsing headings with trailing spaces."""
        text = "# Title   "
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["title"] == "Title"

    def test_multiple_headings(self):
        """Test parsing document with multiple headings."""
        text = """# Main Document
Some content here.
## First Section
More content.
### Subsection
Even more content.
## Second Section
Final content."""
        result = parse_headings(text)
        assert len(result) == 4
        assert result[0] == {"level": 1, "title": "Main Document"}
        assert result[1] == {"level": 2, "title": "First Section"}
        assert result[2] == {"level": 3, "title": "Subsection"}
        assert result[3] == {"level": 2, "title": "Second Section"}

    def test_headings_with_special_characters(self):
        """Test parsing headings containing special characters."""
        text = "# Title with @special #characters & symbols!"
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["title"] == "Title with @special #characters & symbols!"

    def test_headings_with_backticks(self):
        """Test parsing headings with backticks."""
        text = "# `code` in heading"
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["title"] == "`code` in heading"

    def test_headings_with_bold_italic(self):
        """Test parsing headings with markdown formatting."""
        text = "# **Bold** and *italic* text"
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["title"] == "**Bold** and *italic* text"

    def test_invalid_heading_no_space_after_hash(self):
        """Test that # without space is not parsed as heading."""
        text = "#NoSpace"
        result = parse_headings(text)
        assert len(result) == 0

    def test_too_many_hashes(self):
        """Test that more than 6 hashes is not a valid heading."""
        text = "####### Too many hashes"
        result = parse_headings(text)
        assert len(result) == 0

    def test_heading_in_middle_of_line(self):
        """Test that # in middle of line is not parsed as heading."""
        text = "Some text # with hash inside"
        result = parse_headings(text)
        assert len(result) == 0

    def test_heading_with_empty_title(self):
        """Test parsing heading with no title."""
        text = "#"
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["level"] == 1
        assert result[0]["title"] == ""

    def test_heading_with_only_spaces_as_title(self):
        """Test parsing heading with only spaces as title."""
        text = "#    "
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["title"] == ""

    def test_mixed_valid_and_invalid_headings(self):
        """Test document with mix of valid and invalid headings."""
        text = """# Valid Heading
Invalid# heading
## Another valid
###### Sixth level
####### Not valid"""
        result = parse_headings(text)
        assert len(result) == 3
        assert result[0]["title"] == "Valid Heading"
        assert result[1]["title"] == "Another valid"
        assert result[2]["level"] == 6

    def test_windows_line_endings(self):
        """Test parsing text with Windows line endings."""
        text = "# First\r\n## Second\r\n### Third"
        result = parse_headings(text)
        assert len(result) == 3
        assert result[0]["title"] == "First"
        assert result[1]["title"] == "Second"
        assert result[2]["title"] == "Third"

    def test_headings_with_numbers(self):
        """Test parsing headings containing numbers."""
        text = "# Section 1\n## Part 2.5\n### Chapter 3.14159"
        result = parse_headings(text)
        assert len(result) == 3
        assert result[0]["title"] == "Section 1"
        assert result[1]["title"] == "Part 2.5"
        assert result[2]["title"] == "Chapter 3.14159"

    def test_headings_with_unicode(self):
        """Test parsing headings with unicode characters."""
        text = "# Über título\n## 中文标题\n### Заголовок"
        result = parse_headings(text)
        assert len(result) == 3
        assert result[0]["title"] == "Über título"
        assert result[1]["title"] == "中文标题"
        assert result[2]["title"] == "Заголовок"

    def test_headings_with_urls(self):
        """Test parsing headings containing URLs."""
        text = "# Check https://example.com for info"
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["title"] == "Check https://example.com for info"

    def test_long_heading_title(self):
        """Test parsing very long heading titles."""
        long_title = "A" * 200
        text = f"# {long_title}"
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["title"] == long_title

    def test_consecutive_headings(self):
        """Test parsing consecutive headings without content between."""
        text = "# First\n## Second\n### Third"
        result = parse_headings(text)
        assert len(result) == 3
        assert result[0]["level"] == 1
        assert result[1]["level"] == 2
        assert result[2]["level"] == 3

    def test_heading_followed_by_content(self):
        """Test that content after heading is not included."""
        text = "# Title\nThis is content that should be ignored."
        result = parse_headings(text)
        assert len(result) == 1
        assert result[0]["title"] == "Title"

    def test_tabs_before_heading(self):
        """Test that tabs before # prevent heading detection."""
        text = "\t# Indented heading"
        result = parse_headings(text)
        assert len(result) == 0

    def test_spaces_before_heading(self):
        """Test that spaces before # prevent heading detection."""
        text = "  # Indented heading"
        result = parse_headings(text)
        assert len(result) == 0

    def test_return_type_is_list(self):
        """Test that parse_headings always returns a list."""
        result = parse_headings("# Title")
        assert isinstance(result, list)

    def test_return_type_of_elements(self):
        """Test that each element is a dictionary."""
        result = parse_headings("# Title")
        assert isinstance(result[0], dict)

    def test_dictionary_keys(self):
        """Test that returned dictionaries have correct keys."""
        result = parse_headings("# Title")
        assert set(result[0].keys()) == {"level", "title"}

    def test_level_is_integer(self):
        """Test that heading level is an integer."""
        result = parse_headings("# Title")
        assert isinstance(result[0]["level"], int)

    def test_title_is_string(self):
        """Test that heading title is a string."""
        result = parse_headings("# Title")
        assert isinstance(result[0]["title"], str)

    def test_order_preservation(self):
        """Test that headings are returned in document order."""
        text = "# First\n## Second\n# Third\n## Fourth"
        result = parse_headings(text)
        assert result[0]["title"] == "First"
        assert result[1]["title"] == "Second"
        assert result[2]["title"] == "Third"
        assert result[3]["title"] == "Fourth"

    def test_large_document(self):
        """Test parsing a large document with many headings."""
        lines = ["# Main"]
        for i in range(1, 100):
            lines.append(f"## Section {i}")
            lines.append("Some content")
        text = "\n".join(lines)
        result = parse_headings(text)
        assert len(result) == 100
        assert result[0]["level"] == 1
        assert all(h["level"] == 2 for h in result[1:])
