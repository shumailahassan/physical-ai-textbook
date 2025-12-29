#!/bin/bash
# update-textbook-history.sh
# Script to automatically update the textbook history file when new content is generated

HISTORY_FILE="docs/textbook-history.md"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Function to add a new entry to the history file
add_to_history() {
    local module_name="$1"
    local chapter_name="$2"
    local file_path="$3"
    local content_description="$4"

    # Check if the entry already exists
    if grep -q "$file_path" "$HISTORY_FILE"; then
        echo "Entry for $file_path already exists in history."
        return
    fi

    # Create a temporary file with the updated content
    temp_file=$(mktemp)

    # Copy the original file up to the update instructions
    sed '/## Update Instructions/q' "$HISTORY_FILE" > "$temp_file"

    # Add the new entry before the update instructions
    cat >> "$temp_file" << EOL

---

### New Entry Added: $DATE
**Module:** $module_name
**Chapter:** $chapter_name
**File:** $file_path
**Content:**
$content_description

EOL

    # Add back the update instructions
    echo "" >> "$temp_file"
    echo "## Update Instructions" >> "$temp_file"
    echo "This history file should be automatically updated whenever new content is generated for the textbook. All new modules, chapters, code examples, diagrams, and notes should be documented here in chronological order." >> "$temp_file"

    # Replace the original file
    mv "$temp_file" "$HISTORY_FILE"

    echo "Added entry for $file_path to the history file."
}

# Function to scan for new documentation files and add them to history
scan_and_add_new_content() {
    echo "Scanning for new documentation content..."

    # Find all markdown files in docs directory
    find docs/ -name "*.md" -not -path "*/examples/*" | while read -r file; do
        # Skip the history file itself
        if [[ "$file" == "docs/textbook-history.md" ]]; then
            continue
        fi

        # Check if this file is already documented in the history
        if ! grep -q "$file" "$HISTORY_FILE"; then
            # Extract information from the file
            filename=$(basename "$file")
            dirname=$(dirname "$file")

            # Try to determine module and chapter info from filename and content
            module_info="Unknown Module"
            chapter_info="Unknown Chapter"

            # Extract title from frontmatter if available
            title_line=$(grep -m 1 "^title:" "$file" | cut -d '"' -f 2)
            if [[ -z "$title_line" ]]; then
                title_line=$(grep -m 1 "^title:" "$file" | cut -d "'" -f 2)
            fi
            if [[ -z "$title_line" ]]; then
                title_line=$(grep -m 1 "^title:" "$file" | cut -d ' ' -f 2-)
            fi

            if [[ -n "$title_line" ]]; then
                chapter_info="$title_line"
            else
                chapter_info="${filename%.md}"
            fi

            # Determine module based on filename pattern
            if [[ "$filename" == module-0* ]]; then
                module_info="Module 0: Introduction"
            elif [[ "$filename" == module-1* ]]; then
                module_info="Module 1: The Robotic Nervous System (ROS2)"
            elif [[ "$filename" == module-2* ]]; then
                module_info="Module 2: The Digital Twin (Gazebo & Unity)"
            elif [[ "$filename" == module-3* ]]; then
                module_info="Module 3: The AI-Robot Brain (NVIDIA Isaac)"
            elif [[ "$filename" == module-4* ]]; then
                module_info="Module 4: Vision-Language-Action (VLA) Systems"
            elif [[ "$filename" == module-5* ]]; then
                module_info="Module 5: Complete Humanoid Robot Integration"
            fi

            # Extract a brief content description (first 3 lines after title)
            content_desc=$(sed -n '/^# /,$p' "$file" | head -n 10 | grep -v '^#' | head -n 3 | sed 's/^/- /' | sed '/^-\s*$/d' | head -n 3)
            if [[ -z "$content_desc" ]]; then
                content_desc="- Content description not available"
            fi

            # Add to history
            add_to_history "$module_info" "$chapter_info" "$filename" "$content_desc"
        fi
    done
}

# Main execution
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [module_name] [chapter_name] [file_path] [content_description]"
    echo "   Or: $0 --scan (to automatically scan for new content)"
    echo ""
    scan_and_add_new_content
elif [[ "$1" == "--scan" ]]; then
    scan_and_add_new_content
else
    add_to_history "$1" "$2" "$3" "$4"
fi

echo "Textbook history update completed."