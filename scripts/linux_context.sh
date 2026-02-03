#!/bin/bash

# --- Dependencies ---
if ! command -v xclip >/dev/null 2>&1; then
    sudo apt update && sudo apt install xclip -y
fi

# --- Configuration ---
ADDITIONAL_IGNORE=(".git" ".sqlx" "*.log" "target" "node_modules")

# --- Initialisation ---
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$PROJECT_ROOT" || exit 1

TMP_FILE=$(mktemp)

# --- Helper: Check Ignore ---
is_ignored() {
    if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        if git check-ignore -q "$1"; then return 0; fi
    fi
    for pattern in "${ADDITIONAL_IGNORE[@]}"; do
        if [[ "$1" == *"$pattern"* ]]; then return 0; fi
    done
    return 1
}

# --- 1. Process Clipboard ---
# Try to get the "URI list" first (standard for copied files)
RAW_DATA=$(xclip -selection clipboard -o -t text/uri-list 2>/dev/null)

# If URI list is empty, fall back to plain text
if [ -z "$RAW_DATA" ]; then
    RAW_DATA=$(xclip -selection clipboard -o 2>/dev/null)
fi

# Decode URL encoding (like %20) and remove file:// prefix
DECODED_DATA=$(echo "$RAW_DATA" | sed 's/file:\/\///g' | sed 's/\r//g' | perl -pe 's/%([0-9a-f]{2})/chr(hex($1))/eig')

FOUND_ANY=false

# Pre-scan to see if there are actually any valid files before printing headers
while IFS= read -r item; do
    item=$(echo "$item" | xargs)
    if [ -e "$item" ]; then
        if [ "$FOUND_ANY" = false ]; then
            echo "📂 Project root: $PROJECT_ROOT"
            echo "Reading clipboard items..."
            FOUND_ANY=true
        fi

        if [ -d "$item" ]; then
            echo "📁 Processing directory: $item"
            while IFS= read -r subfile; do
                if ! is_ignored "$subfile"; then
                    rel_path=${subfile#$PROJECT_ROOT/}
                    { echo "## File: $rel_path"; echo '```'; cat "$subfile"; echo '```'; echo ""; } >> "$TMP_FILE"
                    echo "  + Added: $rel_path"
                fi
            done < <(find "$item" -type f)
        elif [ -f "$item" ]; then
            if ! is_ignored "$item"; then
                rel_path=${item#$PROJECT_ROOT/}
                { echo "## File: $rel_path"; echo '```'; cat "$item"; echo '```'; echo ""; } >> "$TMP_FILE"
                echo "✔ Added: $rel_path"
            fi
        fi
    fi
done <<< "$DECODED_DATA"

if [ "$FOUND_ANY" = false ]; then
    echo "ℹ️  No files found on clipboard. (Clipboard contains plain text or is empty)"
fi

# --- 2. Folder Structure ---
# If we are in a terminal, we only ask if we actually found files OR if the user wants the tree anyway
read -p "Add folder structure? (y/N): " add_tree
if [[ "$add_tree" =~ ^[Yy]$ ]]; then
    STRICT_TMP=$(mktemp)
    {
        echo "## File & Directory Structure"
        echo '```'
        find . -maxdepth 3 -not -path '*/.*' | sed -e 's/[^-][^\/]*\// |/g' -e 's/|/|-- /g'
        echo '```'
        echo ""
    } > "$STRICT_TMP"
    cat "$TMP_FILE" >> "$STRICT_TMP"
    mv "$STRICT_TMP" "$TMP_FILE"
    echo "✔ Added structure."
fi

# --- 3. Git Diff ---
read -p "Add git diff? (y/N): " add_diff
if [[ "$add_diff" =~ ^[Yy]$ ]]; then
    DIFF=$(git diff)
    if [ -n "$DIFF" ]; then
        { echo "## Git Diff"; echo '```diff'; echo "$DIFF"; echo '```'; } >> "$TMP_FILE"
        echo "✔ Added git diff."
    fi
fi

# --- Finalise ---
if [ -s "$TMP_FILE" ]; then
    cat "$TMP_FILE" | xclip -selection clipboard
    echo -e "\n✅ Success! Context copied to clipboard."
else
    # If the user didn't add files, tree, or diff, don't overwrite their clipboard with nothing
    echo -e "\n⚠️ No content generated. Clipboard preserved."
fi

rm "$TMP_FILE" 2>/dev/null