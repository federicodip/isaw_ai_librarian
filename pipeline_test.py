import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json

# Import functions from pipeline.py
from pipeline import (
    download_pdf_from_html,
    save_pdf_names_to_file,
    load_chunks_from_json,
    txt_to_list_of_strings,
    split_chunk_into_halves
)

class TestPDFProcessing(unittest.TestCase):
    USE_LIVE_URL = False  # Set to True to validate with live data

    def get_test_url(self):
        """
        Returns the appropriate URL based on whether live testing is enabled.
        """
        return "https://dcaa.hosting.nyu.edu/items/show/1804" if self.USE_LIVE_URL else "http://example.com"

    @patch('requests.get')
    def test_download_pdf_from_html_success(self, mock_get):
        """
        Test downloading a PDF when a valid link is found on the page.
        """
        if not self.USE_LIVE_URL:
            # Mock the requests response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = '<html><a href="example.pdf">Download</a></html>'
            mock_get.return_value = mock_response

            # Mock stream response for PDF
            mock_pdf_response = MagicMock()
            mock_pdf_response.status_code = 200
            mock_pdf_response.iter_content = lambda chunk_size: [b'data']
            mock_get.side_effect = [mock_response, mock_pdf_response]

        pdf_names = []
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)

        # Use the appropriate URL
        result = download_pdf_from_html(self.get_test_url(), output_dir, pdf_names)

        if not self.USE_LIVE_URL:
            # Validate mocked behavior
            self.assertTrue(result)
            self.assertEqual(len(pdf_names), 1)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "example.pdf")))
        else:
            # Validate live test behavior (ensure a file is downloaded)
            self.assertTrue(result)
            self.assertGreater(len(pdf_names), 0)

        # Cleanup
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        os.rmdir(output_dir)

    @patch('requests.get')
    def test_download_pdf_from_html_no_pdf_found(self, mock_get):
        """
        Test behavior when no PDF link is found on the page.
        """
        if not self.USE_LIVE_URL:
            # Mock the requests response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = '<html>No PDF here!</html>'
            mock_get.return_value = mock_response

        pdf_names = []
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)

        # Use the appropriate URL
        result = download_pdf_from_html(self.get_test_url(), output_dir, pdf_names)

        if not self.USE_LIVE_URL:
            # Validate mocked behavior
            self.assertFalse(result)
            self.assertEqual(len(pdf_names), 0)
        else:
            # Validate live test behavior (expect no download)
            self.assertFalse(result)

        # Cleanup
        os.rmdir(output_dir)

    def test_save_pdf_names_to_file(self):
        """
        Test saving PDF metadata to a file.
        """
        pdf_names = [{"filename": "example.pdf", "source_url": self.get_test_url()}]
        file_path = "pdf_names.txt"

        save_pdf_names_to_file(pdf_names, file_path)

        self.assertTrue(os.path.exists(file_path))
        with open(file_path, "r") as f:
            content = f.read()
            self.assertIn(f"example.pdf,{self.get_test_url()}", content)

        # Cleanup
        os.remove(file_path)

    @patch("builtins.open", new_callable=mock_open, read_data='[{"content": "test content", "metadata": {"source": "test source"}}]')
    def test_load_chunks_from_json(self, mock_file):
        """
        Test loading chunks from a JSON file.
        """
        chunks = load_chunks_from_json("test.json")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], "test content")
        self.assertEqual(chunks[0]["metadata"]["source"], "test source")

    @patch("builtins.open", new_callable=mock_open, read_data="example.pdf,http://example.com\n")
    def test_txt_to_list_of_strings(self, mock_file):
        """
        Test converting a text file to a list of strings.
        """
        result = txt_to_list_of_strings("test.txt")
        self.assertEqual(result, [["example.pdf", "http://example.com"]])

if __name__ == '__main__':
    unittest.main()
