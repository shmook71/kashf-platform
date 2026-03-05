import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import random

np.random.seed(7)

ENTITIES = ["هيئة الاتصال المرئي والمسموع", " الهيئة السعودية للمدن الصناعية ومناطق التقنية", "هيئة فنون العمارة والتصميم", "(الهيئة السعودية للبيانات والذكاء الاصطناعي (سدايا"]
SERVICES = ["إصدار وثيقة", "تجديد رخصة", "رفع مستند", "حجز موعد"]
STEPS = ["بدء", "تعبئة_نموذج", "رفع_مستند", "مراجعة", "إرسال"]
DEVICES = ["جوال", "كمبيوتر"]
ERRORS = [None, None, None, "E401", "E422", "E500", "EUPLOAD", "ETIMEOUT"]

def new_session_id():
    return str(uuid.uuid4())[:12]  # مجهول

def generate_events(n_sessions=2500, days=14):
    now = datetime.now()
    rows = []

    for _ in range(n_sessions):
        session_id = new_session_id()
        entity = random.choice(ENTITIES)
        service = random.choice(SERVICES)
        device = random.choice(DEVICES)

        start_time = now - timedelta(days=random.randint(0, days-1),
                                     hours=random.randint(0, 23),
                                     minutes=random.randint(0, 59))
        t = start_time

        # نفاذ (تحقق فقط بدون بيانات شخصية)
        rows.append([t, entity, service, session_id, "نفاذ", "تحقق", None, None, device, 1.0])

        dropped = False

        for i, step in enumerate(STEPS):
            # عرض الخطوة
            t += timedelta(seconds=np.random.randint(1, 8))
            rows.append([t, entity, service, session_id, step, "عرض_خطوة", None, None, device, np.nan])

            # نداء API + latency
            latency = int(np.random.lognormal(mean=5.0, sigma=0.4))

            # بطء متعمد في "رفع مستند"
            if service == "رفع مستند" and step == "رفع_مستند":
                latency += np.random.randint(200, 1200)

            t += timedelta(milliseconds=latency)
            rows.append([t, entity, service, session_id, step, "نداء_API", latency, None, device, np.nan])

            # احتمالية خطأ
            err = random.choice(ERRORS)
            if err and np.random.rand() < 0.18:
                rows.append([t, entity, service, session_id, step, "خطأ", latency, err, device, np.nan])

                # إعادة محاولات
                retries = np.random.randint(1, 4)
                for _ in range(retries):
                    t += timedelta(seconds=np.random.randint(2, 10))
                    latency2 = latency + np.random.randint(50, 400)
                    rows.append([t, entity, service, session_id, step, "إعادة_محاولة", latency2, err, device, np.nan])

                # انسحاب بعد خطأ أحيانًا
                if np.random.rand() < 0.35:
                    t += timedelta(seconds=np.random.randint(2, 30))
                    rows.append([t, entity, service, session_id, step, "انسحاب", None, "DROP", device, np.nan])
                    dropped = True
                    break

            # زمن بقاء في الخطوة
            dwell = float(np.random.lognormal(mean=2.6, sigma=0.55))
            if step == "رفع_مستند":
                dwell *= 2.8

            rows.append([t, entity, service, session_id, step, "زمن_خطوة", None, None, device, dwell])

            # انسحاب عشوائي بسيط
            if np.random.rand() < 0.06 and i < len(STEPS)-1:
                t += timedelta(seconds=np.random.randint(2, 40))
                rows.append([t, entity, service, session_id, step, "انسحاب", None, "DROP", device, np.nan])
                dropped = True
                break

        if not dropped:
            t += timedelta(seconds=np.random.randint(1, 8))
            rows.append([t, entity, service, session_id, "إرسال", "نجاح", None, None, device, np.nan])

    df = pd.DataFrame(rows, columns=[
        "timestamp", "entity", "service", "session_id",
        "step", "event_type", "latency_ms", "error_code", "device", "duration_sec"
    ])
    return df.sort_values("timestamp")

if __name__ == "__main__":
    df = generate_events()
    df.to_csv("data/events.csv", index=False)
    print("✅ تم إنشاء data/events.csv")
    print("عدد الصفوف:", len(df))