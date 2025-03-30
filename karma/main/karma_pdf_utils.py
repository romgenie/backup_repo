
    ##########################################################################
    # Helper: PDF reading
    ##########################################################################
    def _read_pdf(self, pdf_path: str) -> str:
        """
        Extract raw text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to read PDF: {pdf_path}, error: {str(e)}")
            return ""

    def _log(self, message: str):
        """
        Simple logger for storing pipeline messages.
        
        Args:
            message: Log message to store
        """
        logger.info(message)
        self.output_log.append(message)