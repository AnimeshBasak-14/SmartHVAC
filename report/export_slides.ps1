# Exports every slide of the presentation to PNG (for checking the layout) and a PDF.
param([string]$Deck = "C:\Users\basak\Desktop\CSC-505\Project\report\HVAC_Service_Planner_Presentation.pptx",
      [string]$OutDir = "C:\Users\basak\AppData\Local\Temp\slides_png")
New-Item -ItemType Directory -Force $OutDir | Out-Null
Get-ChildItem $OutDir -Filter *.png | Remove-Item -Force
$ppt = New-Object -ComObject PowerPoint.Application
try {
    $p = $ppt.Presentations.Open($Deck, $true, $false, $false)
    $p.Export($OutDir, "PNG", 1280, 720)
    $p.SaveAs([System.IO.Path]::ChangeExtension($Deck, ".pdf"), 32)
    "slides: " + $p.Slides.Count
    $p.Close()
} finally {
    $ppt.Quit()
}
