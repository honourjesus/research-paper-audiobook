import fitz  # PyMuPDF
import re
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperStructureAnalyzer:
    """Analyzes research paper structure and extracts different elements"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logger

    def analyze_paper(self, pdf_path: str) -> Dict:
        """
        Analyze paper and identify sections, equations, tables, figures
        
        Returns:
            Dict with structure information
        """
        try
        doc = fitz.open(pdf_path)
        structure = {
            'metadata': self._extract_metadata(doc),
            'sections': [],
            'equations': [],
            'tables': [],
            'figures': [],
            'citations': []
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            blocks = page.get_text("dict")["blocks"]
            
            # Identify sections based on formatting
            sections = self._extract_sections(blocks)
            structure['sections'].extend(sections)
            
            # Extract equations (look for LaTeX patterns)
            equations = self._extract_equations(text, page_num)
            structure['equations'].extend(equations)
            
            # Extract tables
            tables = self._extract_tables(page)
            structure['tables'].extend(tables)
            
            # Extract figures
            figures = self._extract_figures(page, page_num)
            structure['figures'].extend(figures)
            
        doc.close()
        return structure
    
    def _extract_metadata(self, doc) -> Dict:
        """Extract paper metadata"""
        metadata = doc.metadata
        first_page = doc[0].get_text()
        
        # Extract title (usually largest font on first page)
        title = self._extract_title(doc[0])
        
        # Extract authors
        authors = self._extract_authors(first_page)
        
        return {
            'title': title,
            'authors': authors,
            'pdf_metadata': metadata
        }
    
    def _extract_equations(self, text: str, page_num: int) -> List[Dict]:
        """Extract mathematical equations"""
        equations = []
        
        # Pattern for LaTeX equations
        latex_patterns = [
            r'\$\$(.*?)\$\$',  # Display math
            r'\$(.*?)\$',       # Inline math
            r'\\\[(.*?)\\\]',   # LaTeX display
            r'\\\((.*?)\\\)'    # LaTeX inline
        ]
        
        for pattern in latex_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                equations.append({
                    'latex': match.group(1),
                    'page': page_num,
                    'position': match.span()
                })
        
        return equations
    
    def _extract_tables(self, page) -> List[Dict]:
        """Extract table structures"""
        tables = []
        
        # Find table regions (look for grid-like structures)
        text_blocks = page.get_text("dict")["blocks"]
        potential_tables = self._identify_table_regions(text_blocks)
        
        for table_region in potential_tables:
            table_text = page.get_text("text", clip=table_region)
            tables.append({
                'text': table_text,
                'region': table_region,
                'page': page.number
            })
        
        return tables
    
    def _extract_figures(self, page, page_num: int) -> List[Dict]:
        """Extract figure captions and descriptions"""
        figures = []
        
        # Look for "Figure X:" or "Fig. X" patterns
        text = page.get_text()
        figure_pattern = r'(?:Figure|Fig\.?)\s*(\d+)[:\.]\s*(.*?)(?=(?:Figure|Fig\.?|\Z))'
        
        matches = re.finditer(figure_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            figures.append({
                'number': match.group(1),
                'caption': match.group(2).strip(),
                'page': page_num
            })
        
        return figures