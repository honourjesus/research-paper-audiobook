import pandas as pd
from transformer import pipeline
from typing import Dict, List
import logging

class TableSummarizer:
    """Uses ML to generate natural language summaries of tables"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize summarization model
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # CPU, use 0 for GPU
        )
        
        # Initialize table understanding model
        self.table_qa = pipeline(
            "table-question-answering",
            model="google/tapas-base-finetuned-wtq"
        )
    
    def summarize_table(self, table_data: pd.DataFrame, context: str = "") -> Dict:
        """
        Generate comprehensive summary of table
        
        Args:
            table_data: Extracted table as DataFrame
            context: Surrounding text context
            
        Returns:
            Dict with various summaries and statistics
        """
        try:
            # Basic statistics
            stats = self._extract_statistics(table_data)
            
            # Generate natural language summary
            nl_summary = self._generate_narrative(table_data, stats)
            
            # Generate key insights using ML
            insights = self._extract_insights(table_data)
            
            # Identify key relationships
            relationships = self._identify_relationships(table_data)
            
            return {
                'statistics': stats,
                'narrative_summary': nl_summary,
                'key_insights': insights,
                'relationships': relationships,
                'row_count': len(table_data),
                'column_count': len(table_data.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing table: {e}")
            return self._fallback_summary(table_data)
    
    def _extract_statistics(self, df: pd.DataFrame) -> Dict:
        """Extract statistical information from table"""
        stats = {}
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'std': df[col].std()
                }
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                stats[col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None
                }
        
        return stats
    
    def _generate_narrative(self, df: pd.DataFrame, stats: Dict) -> str:
        """Generate narrative description of table"""
        narrative_parts = []
        
        # Overall description
        narrative_parts.append(
            f"This table contains {len(df)} rows and {len(df.columns)} columns."
        )
        
        # Describe each column
        for col in df.columns:
            if col in stats and isinstance(stats[col], dict):
                if 'mean' in stats[col]:
                    narrative_parts.append(
                        f"Column '{col}' has values ranging from "
                        f"{stats[col]['min']:.2f} to {stats[col]['max']:.2f}, "
                        f"with an average of {stats[col]['mean']:.2f}."
                    )
                elif 'unique_values' in stats[col]:
                    narrative_parts.append(
                        f"Column '{col}' contains {stats[col]['unique_values']} "
                        f"unique categories, most commonly '{stats[col]['most_common']}'."
                    )
        
        return " ".join(narrative_parts)
    
    def _extract_insights(self, df: pd.DataFrame) -> List[str]:
        """Extract key insights using ML"""
        insights = []
        
        # Convert table to text for summarization
        table_text = df.to_string()
        
        # Generate summary using BART
        if len(table_text.split()) > 50:  # Only if substantial content
            summary = self.summarizer(
                table_text,
                max_length=100,
                min_length=30,
                do_sample=False
            )
            if summary:
                insights.append(summary[0]['summary_text'])
        
        # Identify trends in numeric data
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            # Check correlations
            correlations = df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = correlations.iloc[i, j]
                    if abs(corr) > 0.7:
                        direction = "positive" if corr > 0 else "negative"
                        insights.append(
                            f"Strong {direction} correlation ({corr:.2f}) between "
                            f"'{numeric_cols[i]}' and '{numeric_cols[j]}'."
                        )
        
        return insights
    
    def _identify_relationships(self, df: pd.DataFrame) -> List[str]:
        """Identify relationships between columns"""
        relationships = []
        
        # Use TAPAS for question answering about relationships
        try:
            questions = [
                "What is the relationship between the columns?",
                "Which column is the most important?",
                "What are the main patterns in this data?"
            ]
            
            for question in questions:
                result = self.table_qa(table=df, query=question)
                if result and 'answer' in result and result['answer']:
                    relationships.append(result['answer'])
                    
        except Exception as e:
            self.logger.warning(f"TAPAS QA failed: {e}")
        
        return relationships
    
    def _fallback_summary(self, df: pd.DataFrame) -> Dict:
        """Fallback summary when ML approaches fail"""
        return {
            'statistics': {},
            'narrative_summary': f"Table with {len(df)} rows and {len(df.columns)} columns",
            'key_insights': [],
            'relationships': [],
            'row_count': len(df),
            'column_count': len(df.columns)
        }