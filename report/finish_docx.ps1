# Updates the table of contents and all page-number fields of the report in Word, saves it and exports a PDF.
param([string]$Doc = "C:\Users\basak\Desktop\CSC-505\Project\report\HVAC_Service_Planner_Report.docx")
$pdf = [System.IO.Path]::ChangeExtension($Doc, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $d = $word.Documents.Open($Doc)
    $d.Repaginate()
    foreach ($i in 1..3) {
        $d.Fields.Update() | Out-Null
        foreach ($t in $d.TablesOfContents) { $t.Update() }
        $d.Repaginate()
    }
    $d.Save()
    $d.ExportAsFixedFormat($pdf, 17)
    "pages: " + $d.ComputeStatistics(2)
    $d.Close(0)
} finally {
    $word.Quit()
}
