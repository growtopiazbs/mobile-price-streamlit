"""Mobile Price: Streamlit-приложение из двух файлов.

Файлы проекта: app.py (интерфейс, прогноз и воспроизводимое обучение)
и model.json (обученная модель, параметры стандартизации и метрики).

Локальный запуск:
    python -m pip install streamlit
    streamlit run app.py

Streamlit Community Cloud:
    Repository: <ваш аккаунт>/mobile-price-streamlit
    Branch: main
    Main file path: app.py
    Python: 3.12

Для работы приложения нужен только Streamlit, который Community Cloud
устанавливает по умолчанию. Поэтому requirements.txt не требуется.
Модель загружается локально; Kaggle, токены и scikit-learn при запуске не нужны.

Повторное обучение (не требуется для запуска):
    python -m pip install scikit-learn==1.8.0
    python app.py --train /path/to/train.csv

train.csv: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


MODEL_PATH = Path(__file__).resolve().with_name("model.json")
FEATURES = ("ram", "battery_power", "px_height", "px_width")
FEATURE_LABELS = (
    "Оперативная память, МБ",
    "Ёмкость аккумулятора, мА·ч",
    "Высота экрана, пиксели",
    "Ширина экрана, пиксели",
)
CLASS_LABELS = ("Бюджетный", "Средний", "Высокий", "Премиальный")
CLASS_DESCRIPTIONS = (
    "Модель относит эту конфигурацию к самому доступному сегменту выборки.",
    "Модель относит эту конфигурацию к среднему ценовому сегменту выборки.",
    "Модель относит эту конфигурацию к высокому ценовому сегменту выборки.",
    "Модель относит эту конфигурацию к самому дорогому сегменту выборки.",
)
COLORS = ("#0F766E", "#2563EB", "#7C3AED", "#B45309")


def load_model(path=MODEL_PATH):
    """Прочитать числовые параметры модели без pickle и scikit-learn."""
    with Path(path).open(encoding="utf-8") as source:
        model = json.load(source)
    if model.get("schema_version") != 1 or model.get("features") != list(FEATURES):
        raise ValueError("Некорректная версия или порядок признаков в model.json")
    if model.get("classes") != [0, 1, 2, 3]:
        raise ValueError("В модели должны быть четыре класса: 0, 1, 2, 3")
    for field in ("mean", "scale", "intercept", "coefficients", "feature_ranges"):
        if len(model[field]) != 4:
            raise ValueError(f"Неверная размерность параметра {field}")
    if any(len(row) != 4 for row in model["coefficients"]):
        raise ValueError("Неверная размерность коэффициентов")
    numeric = model["mean"] + model["scale"] + model["intercept"]
    numeric += [value for row in model["coefficients"] for value in row]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Параметры модели должны быть конечными числами")
    if any(value <= 0 for value in model["scale"]):
        raise ValueError("Масштаб признаков должен быть положительным")
    return model


def predict_proba(model, values):
    """StandardScaler + многоклассовая LogisticRegression + stable softmax.

    Порядок признаков фиксирован в FEATURES. Формулы совпадают с обученной
    sklearn Pipeline; веса и смещения прочитаны из второго файла проекта.
    """
    if len(values) != len(FEATURES) or not all(math.isfinite(x) for x in values):
        raise ValueError("Нужны четыре конечных числовых значения")
    standardized = [
        (value - mean) / scale
        for value, mean, scale in zip(values, model["mean"], model["scale"])
    ]
    scores = [
        bias + sum(weight * value for weight, value in zip(weights, standardized))
        for weights, bias in zip(model["coefficients"], model["intercept"])
    ]
    maximum = max(scores)
    exponents = [math.exp(score - maximum) for score in scores]
    total = sum(exponents)
    return [value / total for value in exponents]


def train_model(csv_path, output_path=MODEL_PATH):
    """Обучить на train.csv Kaggle и сохранить переносимый JSON-артефакт.

    Отложенные 20% данных не участвуют ни в стандартизации, ни в выборе C.
    После оценки модель не переобучается: метрики относятся к сохранённым весам.
    """
    try:
        import numpy as np
        import sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise SystemExit(
            "Для обучения установите scikit-learn: "
            "python -m pip install scikit-learn==1.8.0"
        ) from error

    csv_path = Path(csv_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    x = np.asarray([[float(row[name]) for name in FEATURES] for row in rows])
    y = np.asarray([int(row["price_range"]) for row in rows])
    if not np.isfinite(x).all() or set(y.tolist()) != {0, 1, 2, 3}:
        raise ValueError("Проверьте числовые признаки и классы 0–3 в train.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    search = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
        ]),
        {"classifier__C": [0.1, 1.0, 10.0, 100.0]},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    pipeline = search.best_estimator_
    predicted = pipeline.predict(x_test)
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    model = {
        "schema_version": 1,
        "model_type": "StandardScaler + multinomial LogisticRegression (softmax)",
        "features": list(FEATURES),
        "feature_labels": list(FEATURE_LABELS),
        "classes": classifier.classes_.tolist(),
        "class_labels": list(CLASS_LABELS),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "feature_ranges": [
            {"min": int(x[:, index].min()), "max": int(x[:, index].max())}
            for index in range(len(FEATURES))
        ],
        "dataset": {
            "name": "Mobile Price Classification",
            "author": "Abhishek Sharma (iabhishekofficial)",
            "url": "https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification",
            "download_url": "https://www.kaggle.com/api/v1/datasets/download/iabhishekofficial/mobile-price-classification",
            "file": "train.csv",
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "rows": len(rows),
            "target": "price_range",
            "class_counts": {str(c): int((y == c).sum()) for c in classifier.classes_},
        },
        "training": {
            "sklearn_version": sklearn.__version__,
            "random_state": 42,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "test_size": 0.2,
            "stratified_split": True,
            "cv_folds": 5,
            "c_candidates": [0.1, 1.0, 10.0, 100.0],
            "selected_c": search.best_params_["classifier__C"],
            "cv_accuracy": float(search.best_score_),
            "cv_accuracy_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "test_accuracy": float(accuracy_score(y_test, predicted)),
            "test_macro_f1": float(f1_score(y_test, predicted, average="macro")),
            "test_confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
            "test_used_for_tuning": False,
            "refit_on_all_data": False,
        },
    }
    # Проверяем, что экспорт сохранил прогнозы и вероятности sklearn.
    exported_probabilities = np.asarray([predict_proba(model, row) for row in x_test])
    np.testing.assert_allclose(
        exported_probabilities, pipeline.predict_proba(x_test), atol=1e-12, rtol=1e-12
    )
    Path(output_path).write_text(
        json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(model["training"], ensure_ascii=False, indent=2))
    print(f"Модель сохранена: {output_path}")


def main():
    import streamlit as st

    st.set_page_config(
        page_title="Mobile Price — ценовой сегмент смартфона",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """<style>
        .block-container { max-width: 1160px; padding-top: 2.3rem; }
        .eyebrow { font-size: .76rem; font-weight: 750; letter-spacing: .18em;
                   color: #0F766E; margin-bottom: .65rem; }
        .result-card { padding: 1.6rem 1.7rem; border: 1px solid var(--accent);
                       border-radius: 18px; margin: .5rem 0 1.2rem; }
        .result-kicker { font-size: .83rem; opacity: .72; }
        .result-title { font-size: clamp(1.8rem, 4vw, 2.6rem); font-weight: 750;
                        line-height: 1.2; margin: .55rem 0; color: var(--accent); }
        .result-copy { font-size: .95rem; line-height: 1.6; margin: .7rem 0 0; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
        @media (max-width: 640px) {
            .block-container { padding-top: 1.2rem; }
            .result-card { padding: 1.2rem; }
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">MOBILE PRICE / MACHINE LEARNING</div>', unsafe_allow_html=True)
    st.title("К какому сегменту относится смартфон?")
    st.write(
        "Задайте четыре характеристики — обученная модель определит один "
        "из четырёх ценовых сегментов. Прогноз обновляется автоматически."
    )

    try:
        # Файл небольшой; чтение на каждом rerun также учитывает замену модели.
        model = load_model()
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error("Не удалось загрузить модель. Разместите корректный model.json рядом с app.py.")
        st.caption(str(error))
        st.stop()

    settings_column, result_column = st.columns([1, 1.15], gap="large")
    with settings_column:
        with st.container(border=True):
            st.subheader("Параметры телефона")
            st.caption("Диапазоны ограничены значениями обучающего датасета.")
            defaults = (2048, 1250, 720, 1280)
            helps = (
                "Объём оперативной памяти в мегабайтах. 1 ГБ = 1024 МБ.",
                "Ёмкость батареи в миллиампер-часах.",
                "Вертикальное разрешение экрана. Введите положительное число.",
                "Горизонтальное разрешение экрана.",
            )
            values = []
            for feature, label, bounds, default, help_text in zip(
                FEATURES, FEATURE_LABELS, model["feature_ranges"], defaults, helps
            ):
                minimum = max(1, bounds["min"])
                maximum = bounds["max"]
                values.append(st.slider(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=min(maximum, max(minimum, default)),
                    step=1,
                    key=feature,
                    help=help_text,
                ))

    probabilities = predict_proba(model, values)
    predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
    with result_column:
        st.subheader("Прогноз модели")
        st.markdown(
            f'<div class="result-card" style="--accent:{COLORS[predicted]}">'
            f'<div class="result-kicker">ЦЕНОВОЙ СЕГМЕНТ · КЛАСС {predicted}</div>'
            f'<div class="result-title">{CLASS_LABELS[predicted]}</div>'
            f'<p class="result-copy">{CLASS_DESCRIPTIONS[predicted]}</p></div>',
            unsafe_allow_html=True,
        )
        st.caption("Оценки вероятностей по классам")
        for label, probability in zip(CLASS_LABELS, probabilities):
            st.progress(float(probability), text=f"{label} · {probability:.1%}")
        st.caption(
            "Вероятности рассчитаны моделью и не гарантируют верность отдельного прогноза."
        )

    st.divider()
    metric_accuracy, metric_f1, metric_rows = st.columns(3)
    metrics = model["training"]
    metric_accuracy.metric("Точность на тесте", f'{metrics["test_accuracy"]:.1%}')
    metric_f1.metric("Macro F1 на тесте", f'{metrics["test_macro_f1"]:.3f}')
    metric_rows.metric("Телефонов в датасете", f'{model["dataset"]["rows"]:,}'.replace(",", " "))
    st.caption(
        f'Метрики получены на {metrics["test_rows"]} отложенных телефонах, '
        f'обучение — на {metrics["train_rows"]}. Случайное разбиение: seed=42.'
    )

    with st.expander("О данных и модели"):
        st.markdown(
            "**Датасет:** [Mobile Price Classification на Kaggle]"
            f'({model["dataset"]["url"]}). Целевая переменная — `price_range`.\n\n'
            "**Классы:** 0 — бюджетный, 1 — средний, 2 — высокий, 3 — премиальный.\n\n"
            "**Модель:** стандартизация четырёх признаков и многоклассовая логистическая "
            "регрессия. Коэффициенты обучены на данных и сохранены в `model.json`. "
            "Остальные признаки датасета не используются.\n\n"
            "**Проверка:** 80% данных — обучение, 20% — независимый тест; "
            "разбиение сохраняет доли классов. Параметр C выбран по пяти "
            "кросс-валидационным разбиениям только обучающей части.\n\n"
            "**Ограничения:** датасет содержит относительные ценовые классы, "
            "поэтому приложение не рассчитывает цену в рублях. "
            "Результаты относятся к этому учебному датасету; "
            "они не описывают текущий рынок смартфонов."
        )
        st.caption(
            f'C = {metrics["selected_c"]:g}; средняя точность кросс-валидации: '
            f'{metrics["cv_accuracy"]:.1%}. После оценки сохранены те же веса модели.'
        )
    with st.expander("Как запустить проект"):
        st.markdown(
            "**Локально:** поместите `app.py` и `model.json` в одну папку "
            "и выполните команды ниже."
        )
        st.code("python -m pip install streamlit\nstreamlit run app.py", language="bash")
        st.markdown(
            "**Streamlit Community Cloud:** создайте приложение из GitHub, "
            "выберите репозиторий `mobile-price-streamlit`, ветку `main`, "
            "основной файл `app.py` и Python 3.12. "
            "Два файла достаточны: дополнительных библиотек, ключей и "
            "загрузки датасета при запуске не требуется."
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        parser = argparse.ArgumentParser(description="Обучение Mobile Price модели")
        parser.add_argument("--train", required=True, type=Path, metavar="TRAIN_CSV")
        arguments = parser.parse_args()
        train_model(arguments.train)
    else:
        main()
