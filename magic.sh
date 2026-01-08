find . -type f -name "*.py" ! -path "./.venv/*" -print0 \
| sort -z \
| while IFS= read -r -d '' file; do
    echo "## ${file#./}"
    echo '```python'
    cat "$file"
    echo '```'
    echo
done > all_output.md
