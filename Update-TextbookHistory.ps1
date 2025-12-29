# Update-TextbookHistory.ps1
# PowerShell script to automatically update the textbook history file when new content is generated

param(
    [string]$Module = "",
    [string]$Chapter = "",
    [string]$File = "",
    [string]$Content = "",
    [switch]$Scan = $false
)

$HistoryFile = "docs\textbook-history.md"
$Date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Function to add a new entry to the history file
function Add-ToHistory {
    param(
        [string]$ModuleName,
        [string]$ChapterName,
        [string]$FilePath,
        [string]$ContentDescription
    )

    # Check if the entry already exists
    if (Select-String -Path $HistoryFile -Pattern $FilePath -Quiet) {
        Write-Host "Entry for $FilePath already exists in history."
        return
    }

    # Read the current history file
    $historyContent = Get-Content $HistoryFile

    # Find the line number of the update instructions
    $updateIndex = 0
    for ($i = 0; $i -lt $historyContent.Length; $i++) {
        if ($historyContent[$i] -match "## Update Instructions") {
            $updateIndex = $i
            break
        }
    }

    # Prepare the new entry
    $newEntry = @(
        "",
        "---",
        "",
        "### New Entry Added: $Date",
        "**Module:** $ModuleName",
        "**Chapter:** $ChapterName",
        "**File:** $FilePath",
        "**Content:**",
        $ContentDescription
    )

    # Insert the new entry before the update instructions
    $updatedContent = @()
    $updatedContent += $historyContent[0..($updateIndex-1)]
    $updatedContent += $newEntry
    $updatedContent += @("", "## Update Instructions", "This history file should be automatically updated whenever new content is generated for the textbook. All new modules, chapters, code examples, diagrams, and notes should be documented here in chronological order.")

    # Write the updated content back to the file
    $updatedContent | Set-Content $HistoryFile

    Write-Host "Added entry for $FilePath to the history file."
}

# Function to scan for new documentation files and add them to history
function Scan-AndAddNewContent {
    Write-Host "Scanning for new documentation content..."

    # Find all markdown files in docs directory, excluding examples
    $docFiles = Get-ChildItem -Path "docs" -Filter "*.md" -Recurse | Where-Object { $_.FullName -notlike "*examples*" }

    foreach ($fileInfo in $docFiles) {
        $filePath = $fileInfo.FullName
        $relativePath = Resolve-Path -Path $filePath -Relative
        $fileName = $fileInfo.Name

        # Skip the history file itself
        if ($fileName -eq "textbook-history.md") {
            continue
        }

        # Check if this file is already documented in the history
        if (-not (Select-String -Path $HistoryFile -Pattern $relativePath -Quiet)) {
            # Extract information from the file
            $fileContent = Get-Content $filePath

            # Try to determine module and chapter info from filename and content
            $moduleInfo = "Unknown Module"
            $chapterInfo = "Unknown Chapter"

            # Extract title from frontmatter if available
            $titleLine = ""
            for ($i = 0; $i -lt $fileContent.Length -and $i -lt 10; $i++) {
                if ($fileContent[$i] -match "^title:\s*(.*)") {
                    $titleLine = $matches[1].Trim(' "''')
                    break
                }
            }

            if ([string]::IsNullOrEmpty($titleLine)) {
                $chapterInfo = $fileName -replace '.md$', ''
            } else {
                $chapterInfo = $titleLine
            }

            # Determine module based on filename pattern
            if ($fileName -match "^module-0") {
                $moduleInfo = "Module 0: Introduction"
            } elseif ($fileName -match "^module-1") {
                $moduleInfo = "Module 1: The Robotic Nervous System (ROS2)"
            } elseif ($fileName -match "^module-2") {
                $moduleInfo = "Module 2: The Digital Twin (Gazebo & Unity)"
            } elseif ($fileName -match "^module-3") {
                $moduleInfo = "Module 3: The AI-Robot Brain (NVIDIA Isaac)"
            } elseif ($fileName -match "^module-4") {
                $moduleInfo = "Module 4: Vision-Language-Action (VLA) Systems"
            } elseif ($fileName -match "^module-5") {
                $moduleInfo = "Module 5: Complete Humanoid Robot Integration"
            }

            # Extract a brief content description (first few lines after title)
            $contentDesc = ""
            $inContent = $false
            $descLines = @()

            foreach ($line in $fileContent) {
                if ($line -match "^#") {
                    $inContent = $true
                    continue
                }

                if ($inContent -and $line -notmatch "^#" -and $line -notmatch "^---" -and $line.Trim() -ne "") {
                    $descLines += "- " + $line.Trim()
                    if ($descLines.Length -ge 3) {
                        break
                    }
                }
            }

            if ($descLines.Length -eq 0) {
                $contentDesc = "- Content description not available"
            } else {
                $contentDesc = $descLines -join "`n"
            }

            # Add to history
            Add-ToHistory -ModuleName $moduleInfo -ChapterName $chapterInfo -FilePath $fileName -ContentDescription $contentDesc
        }
    }
}

# Main execution
if ($Scan) {
    Scan-AndAddNewContent
} elseif ($Module -ne "" -and $Chapter -ne "" -and $File -ne "" -and $Content -ne "") {
    Add-ToHistory -ModuleName $Module -ChapterName $Chapter -FilePath $File -ContentDescription $Content
} else {
    Write-Host "Usage:"
    Write-Host "  To add specific entry: .\Update-TextbookHistory.ps1 -Module `"Module 1`" -Chapter `"Chapter 1`" -File `"module-1-example.md`" -Content `"- Description of content`""
    Write-Host "  To scan for new content: .\Update-TextbookHistory.ps1 -Scan"
    Write-Host ""
    Scan-AndAddNewContent
}

Write-Host "Textbook history update completed."