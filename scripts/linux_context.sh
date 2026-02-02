#!/bin/bash

# --- Dependencies ---
if ! command -v xclip >/dev/null 2>&1; then
    echo "Installing xclip..."
    sudo apt update && sudo apt install xclip -y
fi

# --- Configuration ---
ADDITIONAL_IGNORE=(".git" ".sqlx" "*.log" "target" "node_modules")

# --- Initialisation ---
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$PROJECT_ROOT" || exit 1
echo "📂 Project root: $PROJECT_ROOT"

TMP_FILE=$(mktemp)

# --- Helper: Get Clipboard Items (X11 specific) ---
get_clipboard_items() {
    local data=""

    # 1. Try to get URI list (How IDEs/File Managers store file selections)
    # We ask xclip specifically for the 'text/uri-list' target
    data=$(xclip -selection clipboard -o -t text/uri-list 2>/dev/null)

    # 2. If that's empty, try plain text
    if [ -z "$data" ]; then
        data=$(xclip -selection clipboard -o 2>/dev/null)
    fi

    # Clean up: remove file:// prefix and carriage returns
    echo "$data" | sed 's/file:\/\///g' | sed 's/\r//g'
}

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

# --- 1. Process Items ---
# Get data and handle URL decoding (convert %20 to space, etc.)
RAW_DATA=$(get_clipboard_items)

if [ -z "$RAW_DATA" ]; then
    echo "❌ Clipboard appears empty."
    echo "Try copying the files again in RustRover (Ctrl+C)."
    exit 1
fi

# Decode URL-encoded paths (e.g., %20 -> space)
DECODED_DATA=$(echo "$RAW_DATA" | perl -pe 's/%([0-9a-f]{2})/chr(hex($1))/eig')

echo "Reading clipboard items..."
while IFS= read -r item; do
    # Trim whitespace and expand ~ to home
    item=$(echo "$item" | xargs)
    [ -z "$item" ] && continue

    if [ -d "$item" ]; then
        echo "📁 Processing directory: $item"
        while IFS= read -r subfile; do
            if ! is_ignored "$subfile"; then
                rel_path=${subfile#$PROJECT_ROOT/}
                {
                    echo "## File: $rel_path"
                    echo '```'
                    cat "$subfile"
                    echo '```'
                    echo ""
                } >> "$TMP_FILE"
                echo "  + Added: $rel_path"
            fi
        done < <(find "$item" -type f)
    elif [ -f "$item" ]; then
        if ! is_ignored "$item"; then
            rel_path=${item#$PROJECT_ROOT/}
            {
                echo "## File: $rel_path"
                echo '```'
                cat "$item"
                echo '```'
                echo ""
            } >> "$TMP_FILE"
            echo "✔ Added: $rel_path"
        fi
    else
        echo "⚠️  Skipping (path not found): $item"
    fi
done <<< "$DECODED_DATA"

# --- 2. Folder Structure ---
read -p "Add folder structure? (y/N): " add_tree
if [[ "$add_tree" =~ ^[Yy]$ ]]; then
    STRICT_TMP=$(mktemp)
    echo "## File & Directory Structure" > "$STRICT_TMP"
    echo '```' >> "$STRICT_TMP"
    find . -maxdepth 3 -not -path '*/.*' | sed -e 's/[^-][^\/]*\// |/g' -e 's/|/|-- /g' >> "$STRICT_TMP"
    echo '```' >> "$STRICT_TMP"
    echo "" >> "$STRICT_TMP"
    cat "$TMP_FILE" >> "$STRICT_TMP"
    mv "$STRICT_TMP" "$TMP_FILE"
    echo "✔ Added structure."
fi

# --- 3. Git Diff ---
read -p "Add git diff? (y/N): " add_diff
if [[ "$add_diff" =~ ^[Yy]$ ]]; then
    DIFF=$(git diff)
    if [ -n "$DIFF" ]; then
        {
            echo "## Git Diff (Uncommitted Changes)"
            echo '```diff'
            echo "$DIFF"
            echo '```'
        } >> "$TMP_FILE"
        echo "✔ Added git diff."
    fi
fi

# --- Finalise ---
if [ -s "$TMP_FILE" ]; then
    cat "$TMP_FILE" | xclip -selection clipboard
    echo -e "\n✅ Success! Context copied to clipboard."
else
    echo -e "\n⚠️ No content generated."
fi

rm "$TMP_FILE"