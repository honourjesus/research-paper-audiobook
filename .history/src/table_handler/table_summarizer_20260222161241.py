import pandas as pd
from typing import Dict, List
import logging

class TableSummarizer:
    """Generates natural language summaries of tables without heavy ML models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def summarize_table(self, table_data: pd.DataFrame, context: str = "") -> Dict:
        try:
            stats = self._extract_statistics(table_data)
            nl_summary = self._generate_narrative(table_data, stats)
            insights = self._extract_insights(table_data)
            return {
                'statistics': stats,
                'narrative_summary': nl_summary,
                'key_insights': insights,
                'relationships': [],
                'row_count': len(table_data),
                'column_count': len(table_data.columns)
            }
        except Exception as e:
            self.logger.error(f"Error summarizing table: {e}")
            return self._fallback_summary(table_data)

    def _extract_statistics(self, df: pd.DataFrame) -> Dict:
        stats = {}
        for col in df.select_dtypes(include=['number']).columns:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max(),
                'std': df[col].std()
            }
        for col in df.select_dtypes(include=['object']).columns:
            stats[col] = {
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None
            }
        return stats

    def _generate_narrative(self, df: pd.DataFrame, stats: Dict) -> str:
        parts = [f"This table contains {len(df)} rows and {len(df.columns)} columns."]
        for col in df.columns:
            if col in stats:
                if 'mean' in stats[col]:
                    parts.append(
                        f"Column '{col}' ranges from {stats[col]['min']:.2f} to "
                        f"{stats[col]['max']:.2f}, averaging {stats[col]['mean']:.2f}."
                    )
                elif 'unique_values' in stats[col]:
                    parts.append(
                        f"Column '{col}' has {stats[col]['unique_values']} unique values, "
                        f"most commonly '{stats[col]['most_common']}'."
                    )
        return " ".join(parts)

    def _extract_insights(self, df: pd.DataFrame) -> List[str]:
        insights = []
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
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

    def _fallback_summary(self, df: pd.DataFrame) -> Dict:
        return {
            'statistics': {},
            'narrative_summary': f"Table with {len(df)} rows and {len(df.columns)} columns",
            'key_insights': [],
            'relationships': [],
            'row_count': len(df),
            'column_count': len(df.columns)
        }
    