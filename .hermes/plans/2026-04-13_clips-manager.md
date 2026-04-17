# Clips Folder Manager Plan

## Goal
Auto-manage `/mnt/hdd/clips/` — move loose files into game folders, rename clips based on content, keep things tidy.

## Current State
- **Path:** `/mnt/hdd/clips/`
- **Total:** ~2.9GB across 73 clips
- **Folders:** Battlefield6, Fallout4, Ready or Not, Skyrim, Super Battle Golf, Warthunder
- **Loose file:** 1 — `Super Battle Golf - 2026-04-12 9-14-10 PM.mp4` (needs moving into `Super Battle Golf/`)
- **Naming:** Files have auto-generated timestamps. War Thunder clips sometimes have descriptions (e.g., "Target destroyed", "JAS39C"). Most others are just datestamps.
- **Tools available:** ffmpeg, ffprobe (can extract frames/screenshots for vision analysis)

## Naming Convention (proposed)
```
<GameName> - YYYY-MM-DD - <Short Description>.mp4
```
Examples:
- `Super Battle Golf - 2026-04-12 - Hole in one.mp4`
- `War Thunder - 2025-11-29 - JAS39C missile kill.mp4`
- `Battlefield 6 - 2025-10-31 - Tank flank.mp4`

Keep original date from filename. Game name stays consistent per folder.

## What Needs Doing (one-time cleanup)

### Step 1: Move the loose file
- `Super Battle Golf - 2026-04-12 9-14-10 PM.mp4` → `Super Battle Golf/`

### Step 2: Rename existing clips with descriptive names
For each clip:
1. Extract a few frames (beginning, middle, end) using ffmpeg
2. Use vision AI to describe what's happening
3. Generate a short descriptive filename
4. Rename keeping the date

**Priority order (most clips first):**
1. War Thunder — 57 clips (already some have descriptions, but many are just timestamps)
2. Battlefield 6 — 13 clips
3. Super Battle Golf — 4 clips
4. Fallout4 — 3 clips
5. Ready or Not — 1 clip
6. Skyrim — 1 clip

### Step 3: Fix inconsistent folder names
- `Warthunder` → `War Thunder` (match game name / other clips)
- `Fallout4` → `Fallout 4`
- `Battlefield6` → `Battlefield 6`

### Step 4: Save as a cron job (recurring)
Run periodically to:
- Detect new loose files → move to correct folder
- Detect new clips with generic names → extract frames → rename with description
- Handle new games (create folder if game name unrecognized)

## Approach: Frame Extraction for Vision Analysis
```bash
# Extract 3 frames from a clip (start, middle, end)
ffprobe -v error -show_entries format=duration -of csv=p=0 file.mp4
ffmpeg -ss START -i file.mp4 -frames:v 1 frame_start.jpg
ffmpeg -ss MIDDLE -i file.mp4 -frames:v 1 frame_mid.jpg  
ffmpeg -ss END-2 -i file.mp4 -frames:v 1 frame_end.jpg
```
Then send frames to vision_analyze() for description.

## Risks & Tradeoffs
- **Vision accuracy:** AI might misidentify game events. Keep original filenames in a log before renaming.
- **Cost:** 73 clips × 3 frames = ~219 vision calls. Can batch via screenshots saved as files.
- **Speed:** Frame extraction is fast. Vision calls are the bottleneck. Do in batches.
- **New clips in future:** Cron job can handle this automatically.

## Open Questions
- Should Lu approve renames or trust Gleep's judgment? (Suggest: auto-rename, keep a rename log)
- How often should the cron job run? (Suggest: daily or every 6 hours)
- Keep original filenames as metadata somewhere? (Suggest: save a `renames.log` mapping old→new)

## Files Changed
- All clip files in `/mnt/hdd/clips/` (renames and moves)
- New: `/mnt/hdd/clips/.renames.log` (audit trail)
- New: Cron job for ongoing management
