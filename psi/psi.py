def get_cat_buckets(expected_data, actual_data):
    
    """
    Разделяет выборки с дискретными значениями на n_buckets бакетов.
    В случае, когда уникальных числовых значений в выборках меньше, чем n_buckets, вызывает get_cat_buckets()

    Returns:
        buckets - границы бакетов
        expected_bucket_pts - высота каждого бакета в исходной выборке
        actual_bucket_pts - высота каждого бакета в текущей выборке
        expected_data_len - число записей в исходной выборке
        actual_data_len - число записей в текущей выборке
        expected_nans - число пропусков в исходной выборке
        actual_nans - число пропусков в текущей выборке
    """
    
    # Приведение к строковому типу для корректной обработки пропусков
    expected_data = np.array(expected_data).astype(str)
    actual_data = np.array(actual_data).astype(str)

    # Обработка пропусков
    expected_nans = np.count_nonzero(expected_data == 'nan')
    actual_nans = np.count_nonzero(actual_data == 'nan')
    expected_data = expected_data[expected_data != 'nan']
    actual_data = actual_data[actual_data != 'nan']

    # Подсчет количества уникальных значений
    expected_vals, expected_vals_cnt = np.unique(expected_data, return_counts=True)
    actual_vals, actual_vals_cnt = np.unique(actual_data, return_counts=True)
    
    # Объединение уникальных значений из двух выборок
    buckets = np.array(list(set(actual_vals) | set(expected_vals)))

    expected_vals_dict = dict(zip(expected_vals, expected_vals_cnt))
    actual_vals_dict = dict(zip(actual_vals,  actual_vals_cnt))
    
    # 'Заполняем' бакеты
    expected_bucket_pts = np.array([expected_vals_dict.get(val) if val in expected_vals else 0 for val in buckets])
    actual_bucket_pts = np.array([actual_vals_dict.get(val) if val in actual_vals else 0 for val in buckets])
    expected_data_len = np.sum(expected_vals_cnt)
    actual_data_len = np.sum(actual_vals_cnt)

    return buckets, expected_bucket_pts, actual_bucket_pts, expected_data_len, actual_data_len, expected_nans, actual_nans


def get_num_buckets(expected_data, actual_data, n_buckets=10, bucket_type='bins'):
    
    """
    Разделяет числовые выборки на n_buckets бакетов в зависимости от выбранного bucket_type.
    В случае, когда уникальных числовых значений в выборках меньше, чем n_buckets, вызывается get_cat_buckets()

    Returns:
        buckets - границы бакетов
        expected_bucket_pts - высота каждого бакета в исходной выборке
        actual_bucket_pts - высота каждого бакета в текущей выборке
        expected_data_len - число записей в исходной выборке
        actual_data_len - число записей в текущей выборке
        expected_nans - число пропусков в исходной выборке
        actual_nans - число пропусков в текущей выборке
    """

    # Обработка пропусков
    expected_nans = np.count_nonzero(np.isnan(expected_data))
    actual_nans = np.count_nonzero(np.isnan(actual_data))
    expected_data = expected_data[~np.isnan(expected_data)]
    actual_data = actual_data[~np.isnan(actual_data)]
    
    # Подсчет количества уникальных значений в исходных данных
    expected_vals, expected_vals_cnt = np.unique(expected_data, return_counts=True)
    unique_cnt = expected_vals.shape[0]
    
    # Если уникальных значений меньше чем бакетов, то рассчитываем, как для дискретных значений
    if unique_cnt < n_buckets:
        return get_cat_buckets(expected_data, actual_data)

    buckets = np.linspace(0, 1, n_buckets + 1)
    
    # Бакеты равной ширины
    if bucket_type == 'bins':
        unique_vals = np.append(expected_vals, [np.min(actual_data), np.max(actual_data)])
        buckets = np.min(unique_vals) + buckets * (np.max(expected_vals) - np.min(expected_vals))
        
    # Бакеты равной высоты
    elif bucket_type == 'percentiles':
        buckets = np.quantile(expected_data, buckets, method='averaged_inverted_cdf')

    # 'Заполняем' бакеты
    expected_bucket_pts = np.histogram(expected_data, buckets)[0]
    actual_bucket_pts = np.histogram(actual_data, buckets)[0]
    expected_data_len = len(expected_data)
    actual_data_len = len(actual_data)

    # Округление бакетов
    buckets = [round(x, 2) for x in buckets]

    return buckets, expected_bucket_pts, actual_bucket_pts, expected_data_len, actual_data_len, expected_nans, actual_nans


def get_buckets(expected_data, actual_data, n_buckets=10, bucket_type='bins', nan_bucket=True):
    
    """
    Разделяет выборки на бакеты и считает для каждого бакета долю вхождений

    Returns:
        buckets - метки(границы) бакетов
        expected_prc - доля значений в каждом бакете для исходной выборки
        actual_prc - доля значений в каждом бакете для текущей выборки
    """
    
    # Числовые переменные
    if np.issubdtype(np.hstack([expected_data, actual_data]).dtype, np.number):
        buckets, expected_bucket_pts, actual_bucket_pts, expected_data_len, actual_data_len, expected_nans, actual_nans = get_num_buckets(expected_data, actual_data, n_buckets, bucket_type)
    
    # Категориальные переменные
    else:
        buckets, expected_bucket_pts, actual_bucket_pts, expected_data_len, actual_data_len, expected_nans, actual_nans = get_cat_buckets(expected_data, actual_data)

    # Добавление бакета для пропущенных значений
    if nan_bucket and (expected_nans != 0 or actual_nans != 0):
        buckets = np.append(buckets, 'nan')
        expected_bucket_pts = np.append(expected_bucket_pts, expected_nans)
        actual_bucket_pts = np.append(actual_bucket_pts, actual_nans)
        expected_data_len = expected_data_len + expected_nans
        actual_data_len = actual_data_len + actual_nans

    # Расчет процентов
    expected_prc = expected_bucket_pts / expected_data_len
    actual_prc = actual_bucket_pts / actual_data_len

    return buckets, expected_prc, actual_prc


def get_psi(expected_data, actual_data, n_buckets=10, bucket_type='bins', nan_bucket=True):
    """
        Рассчитывает индекс PSI для одной переменной(признака)
        
        Input: 
            expected_data - массив исходных данных
            actual_data - массив текущих данных
            n_buckets - число корзин, на которые делится спектр значений. Не учитывает корзину для пропусков. Не используется при категориальных данных.
            bucket_type - метод разбиения на корзины. 'bins' для корзин равной ширины, 'percentiles' для корзин равной высоты. Не используется при категориальных данных.
            nan_bucket - флаг учета корзины для пропусков
        Output:
            psi - значение индекса
    """
    
    # Поправка для пустых корзин
    EPS = 1e-4

    buckets, expected_prc, actual_prc  = get_buckets(expected_data, actual_data, n_buckets, bucket_type, nan_bucket)
    
    # Обработка пустых корзин
    np.place(expected_prc, expected_prc == 0, EPS)
    np.place(actual_prc, actual_prc == 0, EPS)

    # Расчет индекса
    psi_buckets = (actual_prc - expected_prc) * np.log(actual_prc / expected_prc)
    psi = np.sum(psi_buckets)
    return psi


def get_features_psi(expected_df, actual_df, columns=None, n_buckets=10, bucket_types='bins', nan_buckets=True, returned=False):
    """
        Расчет PSI для всех(или указанных) признаков в датафрейме. Не рассчитывается для столбцов с датами
        Input: 
            expected_df - датафрейм исходных данных
            actual_df - датафрейм текущих данных
            columns - список названий столбцов в исходном датафрейме, которые необходимо учесть. По умолчанию - все
            n_buckets - массив содержащий требуемое число корзин для каждого признака из columns. Не учитывает корзину для пропусков. Не используется при категориальных данных.
            bucket_types - массив содержащий требуемый метод разбиения на корзины для каждого признака из columns. 'bins' для корзин равной ширины, 'percentiles' для корзин равной высоты. Не используется при категориальных данных.
            nan_buckets - bool/массив флагов учета корзины для пропусков для каждого признака из columns
            returned - флаг возвращения списка названий столбцов. Наиболее полезен при columns = None на входе 
        Output:
            features_psi - массив значений индекса для каждого признака
            [columns] - когда returned = True, возвращает названия столбцов, для которых рассчитан индекс
    """
    
    # Обработка пустого списка названий столбцов
    if columns is None:
        columns = expected_df.columns

    features_psi = []

    for i, col in enumerate(columns):
        
        # Обработка столбцов не входящих в исходную выборку
        if col not in expected_df.columns:
            features_psi.append('Not in expected df')

        # Обработка столбцов не входящих в актуальную выборку
        elif col not in actual_df.columns:
            features_psi.append('Not in actual df')
        else:
            
            # Проверка наличия списка n_buckets на входе
            if hasattr(n_buckets, "__len__"):
                n = n_buckets[i]
            else:
                n = n_buckets

            # Проверка наличия списка bucket_types на входе
            if hasattr(bucket_types, "__len__") and type(bucket_types) is not str:
                bucket_type = bucket_types[i]
            else:
                bucket_type = bucket_types

            # Проверка наличия списка nan_buckets на входе
            if hasattr(nan_buckets, "__len__"):
                nan_bucket = nan_buckets[i]
            else:
                nan_bucket = nan_buckets

            psi = get_psi(expected_df[col], actual_df[col], n, bucket_type, nan_bucket)
            features_psi.append(psi)
        
    if returned:
        return features_psi, columns 
    else:
        return features_psi

def get_weighted_psi(expected_df, actual_df, columns=None, weights=None, n_buckets=10, bucket_types='bins', nan_buckets=True, returned=False):
    """
        Расчет PSI методом взвешенного суммирования значений PSI для признаков
        Input: 
            expected_df - датафрейм исходных данных
            actual_df - датафрейм текущих данных
            weights - массив весов для признаков
            columns - список названий колонок в датафрейме, которые необходимо учесть в суммировании. По умолчанию - все
            n_buckets - массив содержащий требуемое число корзин для каждого признака. Не учитывает корзину для пропусков. Не используется при категориальных данных.
            bucket_types - массив содержащий требуемый метод разбиения на корзины для каждого признака. 'bins' для корзин равной ширины, 'percentiles' для корзин равной высоты. Не используется при категориальных данных.
            nan_buckets - bool/массив флагов учета корзины для пропусков
        Output:
            weighted_psi - значение индекса
            [sum_of_weights] - когда returned = True, возвращает сумму весов
    """

    # Обработка пустого списка названий столбцов
    if columns is None:
        columns = expected_df.columns
        
    # Расчет PSI для признаков из списка
    features_psi = get_features_psi(expected_df, actual_df, columns, n_buckets, bucket_types, nan_buckets)

    # Маска корректных значений PSI
    mask = [type(feature_psi) is not str for feature_psi in features_psi]

    # Применение маски к массиву PSI для признаков
    features_psi_masked = np.array(features_psi)[mask]
    features_psi_masked = features_psi_masked.astype(np.float64)
    
    # Обработка пустого списка весов и применение маски
    weights_masked = weights
    if weights is not None:
        weights_masked = np.array(weights)[mask]
    
    # Расчет взвешенной суммы
    weighted_psi = np.average(features_psi_masked, weights=weights_masked, returned=returned)

    return weighted_psi
