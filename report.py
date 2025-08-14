from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from datetime import datetime

class PDFReporter:
    """Generate PDF reports from analysis results"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles without conflicts"""
        
        # Check if custom styles already exist to avoid conflicts
        style_names = [style.name for style in self.styles.byName.values()]
        
        # Only add styles if they don't already exist
        if 'CustomTitle' not in style_names:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            ))
        
        if 'SectionHeader' not in style_names:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceBefore=20,
                spaceAfter=10,
                textColor=colors.darkblue
            ))
        
        if 'CustomBodyText' not in style_names:
            self.styles.add(ParagraphStyle(
                name='CustomBodyText',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceBefore=6,
                spaceAfter=6,
                alignment=TA_LEFT
            ))
        
        if 'CustomCode' not in style_names:
            self.styles.add(ParagraphStyle(
                name='CustomCode',
                parent=self.styles['Normal'],
                fontSize=9,
                fontName='Courier',
                leftIndent=20,
                spaceBefore=10,
                spaceAfter=10,
                backColor=colors.lightgrey
            ))
    
    def generate_report(self, analysis_results):
        """Generate PDF report from analysis results"""
        try:
            # Create buffer
            buffer = io.BytesIO()
            
            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72, leftMargin=72,
                topMargin=72, bottomMargin=18
            )
            
            # Build story
            story = []
            
            # Title
            story.append(Paragraph("DataMind Analysis Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Metadata
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", self.styles['CustomBodyText']))
            story.append(Spacer(1, 20))
            
            # Question
            story.append(Paragraph("Question Asked", self.styles['SectionHeader']))
            question = analysis_results.get('question', 'No question provided')
            story.append(Paragraph(question, self.styles['CustomBodyText']))
            story.append(Spacer(1, 15))
            
            # Answer
            story.append(Paragraph("Key Insight", self.styles['SectionHeader']))
            answer = analysis_results.get('answer', 'No answer available')
            story.append(Paragraph(answer, self.styles['CustomBodyText']))
            story.append(Spacer(1, 15))
            
            # Trust Score
            trust_score = analysis_results.get('trust_score', 0)
            trust_color = self._get_trust_color(trust_score)
            story.append(Paragraph("Trust Score", self.styles['SectionHeader']))
            story.append(Paragraph(
                f"<font color='{trust_color}'>{trust_score}%</font> - {self._get_trust_description(trust_score)}",
                self.styles['CustomBodyText']
            ))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            executive_summary = analysis_results.get('executive_summary', [])
            if executive_summary:
                story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
                for point in executive_summary:
                    story.append(Paragraph(f"• {point}", self.styles['CustomBodyText']))
                story.append(Spacer(1, 15))
            
            # Data Table
            result_df = analysis_results.get('result_df')
            if result_df is not None and not result_df.empty:
                story.append(Paragraph("Results", self.styles['SectionHeader']))
                table = self._dataframe_to_table(result_df)
                if table:
                    story.append(table)
                    story.append(Spacer(1, 15))
            
            # Validation Details
            validation = analysis_results.get('validation', {})
            if validation:
                story.append(Paragraph("Validation Details", self.styles['SectionHeader']))
                
                method = validation.get('validation_method', 'Unknown')
                story.append(Paragraph(f"Validation Method: {method}", self.styles['CustomBodyText']))
                
                issues = validation.get('issues', [])
                if issues:
                    story.append(Paragraph("Issues Found:", self.styles['CustomBodyText']))
                    for issue in issues:
                        story.append(Paragraph(f"• {issue}", self.styles['CustomBodyText']))
                else:
                    story.append(Paragraph("No validation issues found.", self.styles['CustomBodyText']))
                
                story.append(Spacer(1, 15))
            
            # Code Appendix
            code = analysis_results.get('code')
            if code:
                story.append(PageBreak())
                story.append(Paragraph("Generated Code", self.styles['SectionHeader']))
                story.append(Paragraph("Pandas code generated for this analysis:", self.styles['CustomBodyText']))
                
                # Split code into manageable chunks
                code_lines = code.split('\n')
                code_chunks = []
                current_chunk = []
                
                for line in code_lines:
                    if line.strip():
                        # Escape special characters
                        escaped_line = line.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                        current_chunk.append(escaped_line)
                        
                        # Break into chunks of 10 lines
                        if len(current_chunk) >= 10:
                            code_chunks.append('\n'.join(current_chunk))
                            current_chunk = []
                
                # Add remaining lines
                if current_chunk:
                    code_chunks.append('\n'.join(current_chunk))
                
                # Add each chunk as a separate paragraph
                for chunk in code_chunks:
                    if chunk.strip():
                        story.append(Paragraph(f"<pre>{chunk}</pre>", self.styles['CustomCode']))
            
            # Build PDF
            doc.build(story)
            
            # Return bytes
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"PDF generation error: {e}")
            raise Exception(f"PDF generation failed: {str(e)}")
    
    def _get_trust_color(self, trust_score):
        """Get color based on trust score"""
        if trust_score >= 85:
            return 'green'
        elif trust_score >= 60:
            return 'orange' 
        elif trust_score >= 40:
            return 'darkorange'
        else:
            return 'red'
    
    def _get_trust_description(self, trust_score):
        """Get description based on trust score"""
        if trust_score >= 85:
            return "High confidence - Results cross-validated successfully"
        elif trust_score >= 60:
            return "Medium confidence - Minor validation concerns"
        elif trust_score >= 40:
            return "Moderate confidence - Some validation issues detected"
        else:
            return "Low confidence - Significant validation concerns"
    
    def _dataframe_to_table(self, df, max_rows=20):
        """Convert pandas DataFrame to reportlab Table"""
        try:
            # Limit rows
            df_display = df.head(max_rows).copy()
            
            # Convert to string and handle NaN
            df_display = df_display.fillna('N/A').astype(str)
            
            # Create table data
            data = []
            
            # Headers
            headers = list(df_display.columns)
            data.append(headers)
            
            # Rows
            for _, row in df_display.iterrows():
                row_data = []
                for col in headers:
                    value = str(row[col])
                    # Truncate long values
                    if len(value) > 25:
                        value = value[:22] + '...'
                    row_data.append(value)
                data.append(row_data)
            
            # Create table
            table = Table(data)
            
            # Style table
            table.setStyle(TableStyle([
                # Header styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                
                # Body styling
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                
                # Alternating row colors
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.lightgrey])
            ]))
            
            return table
            
        except Exception as e:
            print(f"Table conversion failed: {e}")
            return None