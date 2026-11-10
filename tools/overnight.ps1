# Waits for the main run, then large-crop finetune, then exports + evaluates both checkpoints.
$ErrorActionPreference = "Continue"
Set-Location C:\Projects\mc_ai_challenge
Wait-Process -Id 30288 -ErrorAction SilentlyContinue
$py = ".venv\Scripts\python.exe"
& $py -u train.py --data data/div2k_npy --kodak data/kodak --out runs/ft --resume runs/main/last.pt --reset-opt --steps 280000 --lr 1e-5 --decay-at --crop 512 --batch 4 --eval-every 5000 *> runs\ft\stdout.txt
foreach ($name in @("main", "ft")) {
    & $py tools/export.py runs/$name/last.pt runs/$name/codec.pt *> runs\$name\export.txt
    Copy-Item runs/$name/codec.pt weights/codec.pt -Force
    Remove-Item -Recurse -Force out\$name -ErrorAction SilentlyContinue
    & $py compress.py data/kodak out/$name/compressed *> runs\$name\compress.txt
    & $py decompress.py out/$name/compressed out/$name/reconstructed
    & $py evaluate.py --originals data/kodak --compressed out/$name/compressed --reconstructed out/$name/reconstructed *> runs\$name\final_eval.txt
}
"done" | Out-File runs\ft\OVERNIGHT_DONE.txt
