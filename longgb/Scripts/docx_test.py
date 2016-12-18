#-*- coding:utf-8 -*-
import docx
from docx import Document
from docx.shared import Inches
from docx.shared import RGBColor


document = Document()
table = document.add_table(1, 3)
table.style = 'Light List Accent 1'

list_a = ['Qty', 'SKU', 'Description']
# 标题
heading_cells = table.rows[0].cells
table.columns[0].width = Inches(0.88)
table.columns[1].width = Inches(2.95)
table.columns[2].width = Inches(1.96)

heading_cells[0].text = 'Qty'
heading_cells[1].text = 'SKU'
heading_cells[2].text = 'Description'
# 添加内容
cells = table.add_row().cells
cells[0].text = list_a[0]
cells[1].text = list_a[1]
cells[2].text = list_a[2]

cells = table.add_row().cells
cells[0].text = list_a[0]
cells[1].text = list_a[1]
cells[2].text = list_a[2]

cells = table.add_row().cells
cells[0].text = list_a[0]
cells[1].text = list_a[1]
cells[2].text = list_a[2]

cells = table.add_row().cells
cells[0].text = list_a[0]
cells[1].text = list_a[1]
cells[2].text = list_a[2]


# test the blue
document.save(r'C:\Users\longgb246\Desktop\test\demo.docx')

