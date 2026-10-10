"""Restore source Excel workbooks lost by PPTX import/export; sync edited literal scores."""
from io import BytesIO
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET
from openpyxl import load_workbook
from openpyxl.utils import range_boundaries

ROOT = Path(__file__).resolve().parents[1]
NS = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
source = ROOT.parent / "wt-ppt-latest/ppt/DCC2026_예선_결과정리.pptx"
candidate = ROOT / "outputs/submission_deck/candidate.pptx"
destination = candidate.with_name("candidate_repaired.pptx")
with zipfile.ZipFile(source) as original, zipfile.ZipFile(candidate) as imported:
    files = {name: imported.read(name) for name in imported.namelist()}
    for n in (1, 2, 3):
        chart_name = f"ppt/slides/charts/chart{n}.xml"
        workbook_name = f"ppt/embeddings/Microsoft_Excel_Worksheet{n}.xlsx"
        workbook = load_workbook(BytesIO(original.read(workbook_name)))
        chart = ET.fromstring(files[chart_name])
        updated = 0
        for reference in list(chart.iter(f"{{{NS['c']}}}strRef")) + list(chart.iter(f"{{{NS['c']}}}numRef")):
            formula = reference.find("c:f", NS)
            if formula is None:
                continue
            sheet_name, cell_range = formula.text.rsplit("!", 1)
            sheet = workbook[sheet_name.strip("'")]
            left, top, right, bottom = range_boundaries(cell_range.replace("$", ""))
            coordinates = [(row, col) for row in range(top, bottom+1) for col in range(left, right+1)]
            for point in reference.findall(".//c:pt", NS):
                value = point.find("c:v", NS).text
                row, col = coordinates[int(point.attrib["idx"])]
                if reference.tag.endswith("numRef"):
                    value = float(value)
                sheet.cell(row, col).value = value
                updated += 1
        if not updated:
            raise ValueError(f"No chart cache values restored: {chart_name}")
        payload = BytesIO()
        workbook.save(payload)
        files[workbook_name] = payload.getvalue()
        # Excel serializes floating point cells to its own decimal precision.
        # Refresh the chart's numeric cache from the exact saved worksheet cells.
        with zipfile.ZipFile(BytesIO(payload.getvalue())) as saved_workbook:
            sheet_xml = ET.fromstring(saved_workbook.read("xl/worksheets/sheet1.xml"))
            spreadsheet_ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            saved_values = {cell.attrib["r"]: cell.find("s:v", spreadsheet_ns).text
                            for cell in sheet_xml.findall(".//s:c", spreadsheet_ns)
                            if cell.find("s:v", spreadsheet_ns) is not None}
        from openpyxl.utils import get_column_letter
        for reference in chart.iter(f"{{{NS['c']}}}numRef"):
            formula = reference.find("c:f", NS).text
            left, top, right, bottom = range_boundaries(formula.rsplit("!", 1)[1].replace("$", ""))
            coordinates = [f"{get_column_letter(col)}{row}" for row in range(top, bottom+1) for col in range(left, right+1)]
            for point in reference.findall(".//c:pt", NS):
                point.find("c:v", NS).text = saved_values[coordinates[int(point.attrib["idx"])]]
        external = ET.SubElement(chart, f"{{{NS['c']}}}externalData", {f"{{{NS['r']}}}id": "rId1"})
        ET.SubElement(external, f"{{{NS['c']}}}autoUpdate", {"val": "0"})
        files[chart_name] = ET.tostring(chart, encoding="utf-8", xml_declaration=True)
        relationships = ET.Element(f"{{{REL}}}Relationships")
        ET.SubElement(relationships, f"{{{REL}}}Relationship", {
            "Id": "rId1", "Type": NS["r"] + "/package",
            "Target": f"../../embeddings/Microsoft_Excel_Worksheet{n}.xlsx"})
        ET.register_namespace("", REL)
        files[f"ppt/slides/charts/_rels/chart{n}.xml.rels"] = ET.tostring(relationships, encoding="utf-8", xml_declaration=True)
    content_types = ET.fromstring(files["[Content_Types].xml"])
    if not any(element.attrib.get("Extension") == "xlsx" for element in content_types):
        ET.SubElement(content_types, f"{{{CT}}}Default", {
            "Extension": "xlsx", "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})
    ET.register_namespace("", CT)
    files["[Content_Types].xml"] = ET.tostring(content_types, encoding="utf-8", xml_declaration=True)
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as repaired:
    for name, payload in files.items():
        repaired.writestr(name, payload)
print(f"Source workbook relationships restored; updated literal chart caches: {destination}")
