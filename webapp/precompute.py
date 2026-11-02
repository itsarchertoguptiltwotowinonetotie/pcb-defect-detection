import time
import pcb_core as core
import pipelines as app

t0 = time.time()
settings = []
for i in range(len(app.DEEP)):
    settings += [(app.MODE_SYS, None, app.SRC_DEEP, i, "Motion blur", 4), (app.MODE_SYS, None, app.SRC_DEEP, i, "None", 4),
                 (app.MODE_EXP, app.EXP_ABL, app.SRC_DEEP, i, "None", 4),
                 (app.MODE_EXP, app.EXP_BLUR, app.SRC_DEEP, i, "Motion blur", 4)]
for i in range(len(app.PKU)):
    settings += [(app.MODE_SYS, None, app.SRC_PKU, i, "None", 4), (app.MODE_EXP, app.EXP_PKU, app.SRC_PKU, i, "None", 4)]
for k, (mode, exp, src, i, kind, level) in enumerate(settings):
    h = app.on_run(mode, exp, src, i, None, None, kind, level, 0.5)
    if "went wrong" in h or "not uploaded" in h:
        print("problem:", mode, exp, src, i, h[:300])
    if k % 20 == 0:
        print(f"{k}/{len(settings)} done, {time.time() - t0:.0f}s")
core.save_cache()
print("saved", core.CACHE_FILE, "entries:", len(core._DET), f"in {time.time() - t0:.0f}s")
