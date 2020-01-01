# table-detection
First you need to install the tessract-ocr to your PC: https://github.com/tesseract-ocr/tesseract/wiki

Remember the installation path and replace "pytesseract.pytesseract.tesseract_cmd" in "extract_tabular.py" with your path to tesseract.exe

example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Dependencies
Use "pip install [name]" to install dependencies

opencv-python

openpyxl

pytesseract

jinja2
