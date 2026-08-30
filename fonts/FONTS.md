# Fonts

No font files ship with this repo. `tm-generate` looks for the names below under
`fonts/` (set by `--fonts_dir`) and **silently skips any that are missing**;
`default_languages()` drops a language set only if none of its fonts are present.
So a minimal working setup is just the OFL / Apache faces.

Supply your own copies. Licence status below is best-effort; verify before
redistributing.

## Latin

| file | face | licence | redistributable |
|---|---|---|---|
| `Roboto-Light.ttf` `Roboto-Regular.ttf` `Roboto-Bold.ttf` `Roboto-Italic.ttf` | Roboto | Apache-2.0 | yes |
| `animeace2_bld.ttf` `animeace2_ital.ttf` | Anime Ace 2.0 | Blambot: free for non-commercial, **no redistribution** | no |
| `BlambotClassicBB.ttf` `BlambotClassicBB-Bold.ttf` `BlambotClassicBB-Italic.ttf` | Blambot Classic | Blambot: **no redistribution** | no |

## CJK

| file | face | licence | redistributable |
|---|---|---|---|
| `SourceHanSans.ttc` | Source Han Sans | OFL-1.1 | yes |
| `07YasashisaAntique.otf` `07YasashisaBold.ttf` `YasashisaGothicBold-V2.otf` | Yasashisa Gothic | flopdesign: free incl. commercial | check |
| `NikumaruFont.otf` | 07にくまる | Fontna: free incl. commercial | check |
| `LanobePOPv2.otf` | ラノベPOP | Fontna: free incl. commercial | check |
| `M-NijimiMincho.otf` | にじみ明朝 | Fontna / based on IPA | check |
| `kowaiFont.ttf` | 怖いフォント | free | check |
| `reiko.ttf` | Reiko | public domain (ships with comic-localizer) | yes |
| `msmincho.ttf` | MS Mincho | **proprietary** (Microsoft / Ricoh) | no |

`default_languages()` needs at least one Latin face and, for Japanese/Korean/
Chinese, `SourceHanSans.ttc`.
