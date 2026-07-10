# Eğitim döngüsü hücresi — ACİL düzeltme (autocast/BCE çökmesi)

**Ne yapılacak:** Colab'da açık olan `training/train_colab.ipynb` içindeki
**eğitim döngüsü hücresinin** (Faz (g) — "Eğitim Döngüsü") içeriğini **SİL**,
aşağıdaki kod bloğunun TAMAMINI olduğu gibi o hücreye **yapıştır**, ve
**yalnızca bu hücreyi** yeniden çalıştır.

**ÖNCEKİ HÜCRELERİ TEKRAR ÇALIŞTIRMA** — model, optimizer, veri yükleyiciler
ve checkpoint/resume durumu zaten bellekte; hücreleri baştan çalıştırmak
resume durumunu bozabilir / gereksiz yeniden yükleme yapar.

**Neden çöktü:** `torch.nn.BCELoss()` (gdt kaybı) autocast(bf16) bloğu
içinde çağrılıyordu — PyTorch bunu güvensiz kabul edip hata fırlatıyor.
Aşağıdaki düzeltme, gdt kaybını VE `pix_loss` çağrısını (resmi BiRefNet
`PixLoss`'un içindeki `bce` bileşeni de aynı ham `BCELoss` desenini
kullandığı için) `torch.autocast(..., enabled=False)` ile fp32'ye
yükseltilmiş girdilerle hesaplıyor. Matematik AYNI — yalnızca bu iki kayıp
çağrısı artık fp32'de çalışıyor; model forward'ı (asıl ağır hesap) yine
bf16 autocast altında kalıyor.

```python
import time
import traceback

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
TRAIN_LOG_PATH = STATUS_DIR / "train_log.txt"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

UNITS_PER_HOUR_A100 = 13  # yaklaşık (Colab A100 ~11-13 birim/saat); kesin değeri Colab'ın "Kaynaklar" panelinden doğrulayın.


def log_epoch_row(epoch: int, loss: float, lr_now: float, elapsed_sec: float, eval_mae: float | None) -> None:
    row = f"epoch={epoch}\tloss={loss:.6f}\tlr={lr_now:.8f}\ttime_sec={elapsed_sec:.1f}"
    if eval_mae is not None:
        row += f"\teval_mae={eval_mae:.6f}"
    print(row)
    # Drive'a log yazımı best-effort: geçici bir Drive I/O hatası eğitimi ÖLDÜRMEMELİ
    # (satır konsola zaten basıldı; bir sonraki epoch'un satırı yine denenecek).
    try:
        with open(TRAIN_LOG_PATH, "a") as f:
            f.write(row + "\n")
    except OSError as e:
        print(f"  UYARI: train_log.txt'e yazılamadı ({e}) — eğitim devam ediyor, sonraki epoch'ta tekrar denenecek.")


def save_and_sync_checkpoint(epoch: int) -> None:
    raw_state = model.state_dict()  # torch.compile ise '_orig_mod.' önekli olabilir -- resmi train.py ile AYNI davranış
                                     # (öneki KALDIRMADAN kaydeder); yükleme sırasında her zaman check_state_dict uygulanır.
    payload_out = {
        "model": raw_state,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    # 1) ÖNCE yerel disk — bu her zaman başarılı olmalı (başarısızsa gerçek bir sorun var, hata yükselir).
    local_path = local_ckpt_dir / f"epoch_{epoch}.pth"
    torch.save(payload_out, local_path)
    tcl.prune_old_checkpoints(local_ckpt_dir, KEEP_LAST_N_CHECKPOINTS)

    # 2) SONRA Drive — best-effort: geçici Drive I/O hatası (kota/senkron takılması)
    # eğitimi öldürmez; yerel kopya güvende, bir SONRAKİ epoch yeni bir Drive
    # kopyası deneyecek (en kötü durumda Drive 1 epoch geriden gelir).
    try:
        drive_path = drive_ckpt_dir / f"epoch_{epoch}.pth"
        shutil.copy2(local_path, drive_path)
        tcl.prune_old_checkpoints(drive_ckpt_dir, KEEP_LAST_N_CHECKPOINTS)
        print(f"  checkpoint kaydedildi + Drive'a kopyalandı: {drive_path}")
    except OSError as e:
        print(f"  UYARI: checkpoint Drive'a kopyalanamadı ({e}) — YEREL kopya güvende: {local_path}; "
              f"sonraki epoch'ta yeniden denenecek.")


def train_one_epoch(epoch: int) -> float:
    model.train()

    # --- Fine-tune hilesi: mutlak epoch numarasına göre TABANDAN yeniden hesapla (resume-güvenli, bkz. yukarıdaki not) ---
    pix_loss.lambdas_pix_last = dict(BASE_LAMBDAS_PIX_LAST)
    if tcl.should_apply_finetune_reweight(epoch, EPOCHS, config.finetune_last_epochs):
        n = epoch - (EPOCHS + config.finetune_last_epochs)
        if config.task == "Matting":
            pix_loss.lambdas_pix_last["mse"] = BASE_LAMBDAS_PIX_LAST["mse"] * (0.9 ** n)
            pix_loss.lambdas_pix_last["ssim"] = BASE_LAMBDAS_PIX_LAST["ssim"] * (0.9 ** n)
        else:
            pix_loss.lambdas_pix_last["bce"] = BASE_LAMBDAS_PIX_LAST["bce"] * 0
            pix_loss.lambdas_pix_last["iou"] = BASE_LAMBDAS_PIX_LAST["iou"] * (0.5 ** n)
            pix_loss.lambdas_pix_last["mae"] = BASE_LAMBDAS_PIX_LAST["mae"] * (0.9 ** n)

    running_sum, running_n = 0.0, 0
    n_batches = len(train_loader)
    optimizer.zero_grad()
    for micro_step, batch in enumerate(train_loader):
        inputs = batch[0].to(device, non_blocking=True)
        gts = batch[1].to(device, non_blocking=True)
        class_labels = batch[2].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            scaled_preds, class_preds_lst = model(inputs)
            if config.out_ref:
                (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds

        # neden: BCE (BCELoss / binary_cross_entropy) autocast altında yasak; resmi BiRefNet
        # train.py/loss.py ile AYNI matematik, sadece bu blok (gdt + pix_loss) fp32’de hesaplanıyor.
        with torch.autocast(device_type="cuda", enabled=False):
            if config.out_ref:
                loss_gdt = None
                for gi, (gp, gl) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                    gp_i = torch.nn.functional.interpolate(gp.float(), size=gl.shape[2:], mode="bilinear", align_corners=True).sigmoid()
                    gl_i = gl.float().sigmoid()
                    li = criterion_gdt(gp_i, gl_i)
                    loss_gdt = li if gi == 0 else loss_gdt + li
            loss_cls = 0.0 if None in class_preds_lst else cls_loss(class_preds_lst, class_labels)
            # pix_loss (resmi PixLoss) da ’bce’ bileşeninde ham BCELoss kullanıyor -- aynı nedenle fp32’ye yükseltiliyor.
            scaled_preds_f = [sp.float() for sp in scaled_preds]
            loss_pix, _loss_dict_pix = pix_loss(scaled_preds_f, torch.clamp(gts, 0, 1).float(), pix_loss_lambda=1.0)
            loss = loss_pix + loss_cls
            if config.out_ref:
                loss = loss + loss_gdt * 1.0

        (loss / ACCUM).backward()
        if (micro_step + 1) % ACCUM == 0 or (micro_step + 1) == n_batches:
            optimizer.step()
            optimizer.zero_grad()

        running_sum += loss.item() * inputs.size(0)
        running_n += inputs.size(0)
        if micro_step % 200 == 0:
            print(f"  epoch {epoch} iter {micro_step}/{n_batches} loss={loss.item():.5g}")

    lr_scheduler.step()
    return running_sum / max(running_n, 1)


def main() -> None:
    for epoch in range(epoch_st, EPOCHS + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(epoch)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        eval_mae = None
        if epoch % N_EVAL_EVERY == 0 or epoch == EPOCHS:
            eval_mae = run_quick_eval(model, EVAL_STEMS, local_val_im, local_val_gt, device)

        log_epoch_row(epoch, avg_loss, current_lr, elapsed, eval_mae)
        save_and_sync_checkpoint(epoch)

        # ÖLÇÜLMÜŞ maliyet raporu (parametre hücresindeki teorik tabloyla karşılaştırın):
        hours = elapsed / 3600
        est_units = hours * UNITS_PER_HOUR_A100
        remaining = EPOCHS - epoch
        print(f"  MALİYET: bu epoch {hours:.2f} saat ≈ {est_units:.0f} birim "
              f"(A100 ~{UNITS_PER_HOUR_A100} birim/saat varsayımıyla); "
              f"kalan {remaining} epoch ≈ {remaining * hours:.1f} saat ≈ {remaining * est_units:.0f} birim. "
              f"Bütçenizi aşacaksa şimdi durdurun — RESUME='auto' kaldığı yerden devam eder.")

    print("EĞİTİM TAMAMLANDI.")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    print(tb)
    try:  # FATAL kaydı da best-effort — Drive yazılamıyorsa asıl hatayı gölgelemesin.
        with open(TRAIN_LOG_PATH, "a") as f:
            f.write(f"epoch=FATAL\ttraceback={tb!r}\n")
    except OSError as log_err:
        print(f"UYARI: FATAL kaydı train_log.txt'e yazılamadı ({log_err}).")
    raise
```
