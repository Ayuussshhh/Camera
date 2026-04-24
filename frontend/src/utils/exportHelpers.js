import * as XLSX from "xlsx";
import jsPDF from "jspdf";
import "jspdf-autotable";

export function exportRowsToCsv(fileName, rows) {
  if (!rows.length) {
    return;
  }

  const headers = Object.keys(rows[0]);
  const csvContent = [
    headers.join(","),
    ...rows.map((row) =>
      headers
        .map((header) => `"${(row[header] ?? "").toString().replace(/"/g, '""')}"`)
        .join(",")
    )
  ].join("\n");

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = `${fileName}.csv`;
  link.click();

  URL.revokeObjectURL(url);
}

export function exportRowsToExcel(fileName, rows, sheetName = "Attendance") {
  const worksheet = XLSX.utils.json_to_sheet(rows);
  const workbook = XLSX.utils.book_new();

  XLSX.utils.book_append_sheet(workbook, worksheet, sheetName);
  XLSX.writeFile(workbook, `${fileName}.xlsx`);
}

export function exportRowsToPdf(fileName, title, columns, rows) {
  const doc = new jsPDF();

  doc.setFontSize(16);
  doc.text(title, 14, 18);

  doc.autoTable({
    startY: 26,
    head: [columns],
    body: rows.map((row) => columns.map((column) => row[column] ?? "")),
    styles: {
      fontSize: 9
    },
    headStyles: {
      fillColor: [15, 118, 110]
    }
  });

  doc.save(`${fileName}.pdf`);
}
