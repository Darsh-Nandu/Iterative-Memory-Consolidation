# Open: src/arxiv_fetcher.py
import os
import arxiv
import urllib.request  # <--- ADD THIS IMPORT AT THE TOP
from pypdf import PdfReader
from loguru import logger

class ArxivDatasetFetcher:
    def __init__(self, storage_dir: str = "datasets"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.client = arxiv.Client()

    def fetch_and_extract(self, search_query: str, max_results: int = 1) -> str:
        logger.info(f"Querying arXiv API for: '{search_query}' (Limit: {max_results})")
        
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = list(self.client.results(search))
        if not results:
            raise RuntimeError(f"Zero search entries returned from arXiv for query: '{search_query}'")
            
        paper = results[0]
        logger.info(f"Found Paper Target: {paper.title} (ID: {paper.get_short_id()})")
        
        safe_title = "".join([c if c.isalnum() else "_" for c in paper.title[:30]]).lower()
        pdf_filename = f"{safe_title}.pdf"
        txt_filename = f"{safe_title}.txt"
        
        pdf_path = os.path.join(self.storage_dir, pdf_filename)
        txt_path = os.path.join(self.storage_dir, txt_filename)
        
        # Step 1: Download Paper PDF via native urllib
        logger.info(f"Downloading vector asset PDF to: {pdf_path}")
        # CHANGE THIS LINE: Use urlretrieve instead of paper.download_pdf
        urllib.request.urlretrieve(paper.pdf_url, pdf_path) 
        
        # Step 2: Extract text from PDF layout via pypdf
        logger.info(f"Extracting string tokens from PDF structural content layout...")
        extracted_text_blocks = []
        
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text_blocks.append(page_text)
                    
            full_document_text = "\n".join(extracted_text_blocks)
            
            with open(txt_path, "w", encoding="utf-8") as text_file:
                text_file.write(full_document_text)
                
            logger.info(f"Successfully serialized raw text corpus to: {txt_path}")
            
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                
            return full_document_text
            
        except Exception as e:
            logger.error(f"Failed to process and parse downloaded PDF matrix structure: {e}")
            raise e