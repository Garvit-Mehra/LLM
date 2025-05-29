import streamlit as st
import camelot
import PyPDF2
import os
import pandas as pd
import fitz  # PyMuPDF
import base64
import io
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class PDFProcessor:
    def __init__(self):
        self.temp_file_path = "temp_pdf.pdf"
        self.no_image_pdf_path = "temp_no_images.pdf"

    def extract_text(self, file):
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    def merge_multiline_header(self, df):
        header_rows = []
        data_start_idx = 0
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            text_ratio = sum(cell.isalpha() or cell.isspace() for cell in ''.join(row))
            total_len = sum(len(str(cell)) for cell in row)
            if text_ratio > 0.6 * total_len:
                header_rows.append(row)
                data_start_idx = i + 1
            else:
                break
        if len(header_rows) > 1:
            merged_header = [" ".join(filter(None, col)).strip() for col in zip(*header_rows)]
        elif header_rows:
            merged_header = header_rows[0].tolist()
        else:
            merged_header = df.columns.tolist()
        df.columns = merged_header
        df = df.iloc[data_start_idx:].reset_index(drop=True)
        return df

    def clean_multiline_rows(self, df):
        cleaned_rows = []
        current_row = [""] * len(df.columns)
        for _, row in df.iterrows():
            non_empty = sum(bool(str(cell).strip()) for cell in row)
            if non_empty >= len(df.columns) // 2:
                if any(current_row):
                    cleaned_rows.append(current_row)
                current_row = row.tolist()
            else:
                for i, cell in enumerate(row):
                    if cell:
                        current_row[i] = f"{current_row[i]} {cell}".strip()
        if any(current_row):
            cleaned_rows.append(current_row)
        return pd.DataFrame(cleaned_rows, columns=df.columns)

    def gpt_describe_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image from the PDF:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}",
                            }
                        },
                    ],
                }
            ],
            max_tokens=300
        )

        return response.choices[0].message.content

    def extract_image_descriptions(self, pdf_path):
        doc = fitz.open(pdf_path)
        descriptions = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    description = self.gpt_describe_image(image_bytes)
                    descriptions.append(f"Page {page_num + 1} Image: {description}")
                except Exception as e:
                    descriptions.append(f"Page {page_num + 1} Image: Error describing image ({e})")
        return "\n".join(descriptions)

    def remove_images_and_save(self, pdf_path, output_path):
        doc = fitz.open(pdf_path)
        for page in doc:
            img_rects = []
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    for img_info in page.get_image_info(xref):
                        img_rects.append(img_info["bbox"])
                except:
                    pass
            for rect in img_rects:
                page.add_redact_annot(rect, fill=(1,1,1))
            page.apply_redactions()
        doc.save(output_path)

    def extract_tables_and_text(self, file):
        try:
            file.seek(0)
            with open(self.temp_file_path, "wb") as f:
                f.write(file.read())

            self.remove_images_and_save(self.temp_file_path, self.no_image_pdf_path)
            image_descriptions = self.extract_image_descriptions(self.temp_file_path)

            with open(self.no_image_pdf_path, "rb") as f:
                pdf_text = self.extract_text(f)

            tables = camelot.read_pdf(
                self.no_image_pdf_path,
                pages="all",
                flavor="lattice",
                split_text=True,
                strip_text='\n'
            )

            if not tables or tables.n == 0:
                st.warning("No tables detected in lattice mode. Trying stream mode...")
                tables = camelot.read_pdf(
                    self.no_image_pdf_path,
                    pages="all",
                    flavor="stream",
                    row_tol=15
                )

            if not tables or tables.n == 0:
                st.warning("No tables detected.")
                return [{"combined_content": f"Image Descriptions:\n{image_descriptions}"}], pdf_text

            table_data = []
            for table in tables:
                df = table.df
                df = self.merge_multiline_header(df)
                df = self.clean_multiline_rows(df)
                table_markdown = df.to_markdown(index=False)
                combined_content = (
                    f"Table (Page {table.page}):\n{table_markdown}\n\n"
                    f"Image Descriptions:\n{image_descriptions}\n\n"
                    f"PDF Text:\n{pdf_text}"
                )
                table_data.append({
                    "markdown": table_markdown,
                    "combined_content": combined_content,
                    "page": table.page
                })

            return table_data, pdf_text

        except Exception as e:
            st.error(f"Error extracting tables: {str(e)}")
            return [], ""
        finally:
            for path in [self.temp_file_path, self.no_image_pdf_path]:
                if os.path.exists(path):
                    os.remove(path)
                    