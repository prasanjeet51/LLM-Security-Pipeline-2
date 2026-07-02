param([string]$Command = $env:CLAUDE_TOOL_INPUT)
if ($Command -match "rm -rf.*(models[/\\]|data[/\\]processed|mlruns)") {
    Write-Error "BLOCKED: destructive command on protected folder"
    exit 1
}
exit 0
