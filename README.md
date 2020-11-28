# Метрики в задаче классификации

Поработаем с [набором медицинских данных](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) об параметрах опухоли на рентгеновском снимке.
 1. Построим классификатор на основе логистической регрессии (класс [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))
 2. Построим классификатор на основе метод k ближайших соседей (класс [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html))
 3. Сравним характеристики точности, полноты и F1 оценки для двх моделей (функция [precision_recall_fscore_support](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html))
 4. Построим для двух моделей PR-кривую и вычислите площадь по кривой (функция [precision_recall_curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve))
 5. Построим для двух моделей ROC-кривую и вычислите площадь под ней (функция [roc_curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve))
