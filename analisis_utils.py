import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from ipywidgets import widgets
from IPython.display import display

import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import FigureWidget
from plotly.subplots import make_subplots
import plotly.io as pio

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, FixedTicker, BooleanFilter, CDSView, HoverTool, Select
from bokeh.io import output_notebook, curdoc
from bokeh.layouts import gridplot, column
from bokeh.transform import linear_cmap
from bokeh.palettes import magma, tol

from seaborn import heatmap

from phik import resources, phik_matrix
from phik.binning import bin_data
from phik.report import plot_correlation_matrix

from umap import UMAP
import openTSNE
from sklearn.decomposition import PCA
from time import time as tm

def calc_recall(true_labels, pred_labels, k, exclude_self=False, return_mistakes=False):
    '''
    счиатет recall@k для приближенного поиска соседей
    
    true_labels: np.array (n_samples, k)
    pred_labels: np.array (n_samples, k)
    
    exclude_self: bool
        Если query_data была в трейне, считаем recall по k ближайшим соседям, не считая самого себя
    return_mistakes: bool
        Возвращать ли ошибки
    
    returns:
        recall@k
        mistakes: np.array (n_samples, ) с количеством ошибок
    '''
    n = true_labels.shape[0]
    n_success = []
    shift = int(exclude_self)
    
    for i in range(n):
        n_success.append(np.intersect1d(true_labels[i, shift:k+shift], pred_labels[i, shift:k+shift]).shape[0])
        
    recall = sum(n_success) / n / k
    if return_mistakes:
        mistakes = k - np.array(n_success)
        return recall, mistakes
    return recall

def plot_ann_performance(build_data, 
                         query_data, 
                         index_dict, 
                         k, 
                         flat_build_func, 
                         flat_search_func, 
                         query_in_train=False, # пока не используется
                         qps_line=None, 
                         recall_line=None, 
                         title=None ):
    '''
    Функция для оценки качества работы алгоритмов поиска соседей. В основном это нужно для библиотек faiss и hnsw
    build_data - данные, на которых будут строиться индексы подаваемых на вход алгоритмов
    query_data - тестовые данные, на которых будет проверяться качество алгоритмов
    index_dict - "умный" словарь, в котором хранятся параметры для моделей. Схема построения ниже
    k - int. количество соседей в поиске
    flat_build_func - функция, по которой будут строиться точные индексы для точных ответов
    flat_search_func - функция, которая точно находит соседей по индексам flat_build_func
    query_in_train=False - bool. Условие, по которым мы указываем, находятся ли тестовые данные в тренеровочной выборке
    qps_line=None - float. Если указано, нарисуем горизонтальную линию по этому значению
    recall_line=None - float. Если указано, нарисуем вертикальную линию по этому значению
    title=None - string. Название заголовка графика
    
    Пример index_dict:
    index_dict = {
    'faiss IndexIVFPQ' : {
        'fixed_params' : {
            'dim' : dim,
            'coarse_index' : faiss.IndexFlatL2(dim),
            'nlist' : 16*8,
            'M': 5,
            'nbits' : 8,
            'metric': faiss.METRIC_L2
        },
        'build_func' : build_IVFPQ,
        'search_param': ('n_probe', [1, 10, 50, 70, 100]),
        'search_func': search_faiss
    },
    'hnsw M=32, efC=32' : {
        'fixed_params': {
            'dim': dim, 
            'space': 'l2', 
            'M': 32, 
            'ef_construction': 32},
        'build_func': build_hnsw, # ест build_data и fixed_params, возвращает построенный индекс
        'search_param': ('efSearch', [10, 20, 64, 100]), # (имя параметра поиска, [используемые значения])
        'search_func': search_hnsw # ест index, query_data, k_neighbors, search_param, возвращает distances, labels
    }
    }
    '''
    model_time = []
    model_qps = dict()
    model_recall = dict()
    
    mosaic = '''
    SCC
    '''
    fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(12, 6))
    
    
    # Рассчёт правильных ответов
    true_index, flat_time1 = flat_build_func()
    _, true_labels, flat_time2 = flat_search_func(true_index, query_data, k)
    flat_time = flat_time1*20 + flat_time2
    
    # Прогоняем каждую модель через алгоритм
    for model in index_dict:
        model_qps[model] = []
        model_recall[model] = []
        
        index, time_build = index_dict[model]['build_func'](build_data, **index_dict[model]['fixed_params'])
        
        time_i=[]
        param_name, param_value = index_dict[model]['search_param']
        for i, value in enumerate(param_value):
            _, pred_labels, time = index_dict[model]['search_func'](index, query_data, k, value)
            
            time_i.append(time+time_build)
            qps_i = len(query_data) / (time)

            model_qps[model].append(qps_i)
            model_recall[model].append(calc_recall(true_labels, pred_labels, k))
        
        model_time.append(np.mean(time_i))
    
    # Рисуем графики  

    for i, model in enumerate(index_dict.keys()):
        ax['S'].bar(model, model_time[i])
    
    ax['S'].tick_params(axis='x', labelrotation=90)
    ax['S'].set_ylabel('time, sec')
    ax['S'].grid()
    

    for model in index_dict.keys():
        param_name, param_value = index_dict[model]['search_param']
        ax['C'].plot(model_recall[model], model_qps[model], '-o', label=model)
        for i, value in enumerate(param_value):
            ax['C'].annotate(f'{param_name}={value}', (model_recall[model][i], model_qps[model][i]))
    
    ax['C'].axhline(y=len(query_data)/flat_time, color='r', linestyle='--')
    ax['C'].set_yscale('log')
    ax['C'].set_xlabel(f'recall@{k}')
    ax['C'].legend()
    ax['C'].set_title(title)
    if qps_line:
        ax['C'].axhline(y=qps_line, color='b', linestyle='--')
    if recall_line:
        ax['C'].axvline(x=recall_line, color='b', linestyle='--')
        
    ax['C'].grid()
    plt.show()

def feature_distributions(data: pd.DataFrame, hue: str, title: str=''):
    """
    Отрисовываем распределение каждому признаку относительно целевого признака hue (в нём должно быть не более 5 уникальных значений).
    
    Args:
    data - таблица, по которой строются графики
    hue - название целевого признака
    title - заголовок графика
    
    Return:
    fig - график plotly, который мы подготовили
    """
    columns = list(data.drop(hue, axis=1).columns)
    fig = make_subplots(rows=int(len(columns)/2)+1, cols=2, subplot_titles=columns)

    count = 0
    hue_names = data[hue].unique()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    for row in range(int((len(columns))/2)): 
        for col in range(2):

            for i, h in enumerate(hue_names):
                fig.add_trace(go.Histogram(x=data[data[hue]== h][columns[count]],
                                        name=f'{h}',
                                        marker_color=colors[i],
                                        histnorm='probability'),
                                        row=row + 1, col=col + 1)
            count +=1 #UwU
    fig.update_layout(height=300*int((len(columns)-1)/2), width=1000, title_text=title, showlegend=False)

    return fig

def compose_feature(before: pd.DataFrame, after: pd.DataFrame, feature: str, bin_hue: str, height: int=300):
    '''
    Рисовалка, которая отслеживает как изменяется распределение признака feature по целевому признаку bin_hue до и после преобразования 
    (пределать под любую целевую переменную bin_hue)

    Args:
    before - таблица до изменения feature
    after - таблица после изменения feature
    featute - название изменяемого признака. Должен находиться в таблицах before и after
    bin_hue - название целевого признака, который может принимать значения 0 и 1

    Return:
    fig - plotly график распределения
    '''
    fig = make_subplots(rows=1, cols=2, subplot_titles=['before', 'after'])
    fig.add_trace(go.Histogram(x=before[before[bin_hue]==0][feature],
                               name='0',
                               marker_color='#08D9D6',
                               histnorm='probability'),
                               row=1, col=1)
    fig.add_trace(go.Histogram(x=before[before[bin_hue]==1][feature],
                               name='1',
                               marker_color='#E32636',
                               histnorm='probability'),
                               row=1, col=1)
    
    fig.add_trace(go.Histogram(x=after[after[bin_hue]==0][feature],
                               name='0',
                               marker_color='#08D9D6',
                               histnorm='probability'),
                               row=1, col=2)
    fig.add_trace(go.Histogram(x=after[after[bin_hue]==1][feature],
                               name='1',
                               marker_color='#E32636',
                               histnorm='probability'),
                               row=1, col=2)
    
    fig.update_layout(height=height, title_text=feature, showlegend=False)
    return fig

def get_df_info(df: pd.DataFrame, thr: float=0.8):
    """
    Создаёт краткую информации о таблице по каждому столбцу. В качестве индексов
    выступают названия колонок df. Крайне не рекомендуется использовать эту таблицу
    для считывания данных, т.к. в ней могут использоваться спец. символы для красоты,
    которые трудно обнаружить!
    Столбцы результирующей таблицы:
    dtype : тип колонки
    exemples : случайные 2 примера из таблицы (не NaN)
    vc_max_el : самый частый элемент в колонке
    vc_max_freq : доля самого частого элемента в колонке
    nunique : количество уникальных элементов в колонке (включая NaN)
    nan : доля NaN в колонке ('-1' в колонке отсутвуют Nan)
    zero : доля нулей в колонке ('-1' в колонке отсутствуют нули)
    empty_str : количество пустых строк в колонке ('-1' в колонке отсутствуют пустые строки)
    trash_score : показатель "ипорченности данных"
      trash_score = max[(суммарная доля nan, zero, empty_str), (доля vc_max_freq, если она больше порога thr, иначе 0)]

    Args:
    df (pandas.DataFrame) : таблица, данные которой мы хотим изучить
    thr (float) : порог для trash_score, после которого мы будем присваивать
                  колонке значение доли самого частого элемента в колонке, если
                  значение окажется больше чем суммарная доля nan, zero, empty_str
                  изменяется в диапозоне (0, 1). 

    Returns:
    info (pandas.DataFrame) : таблица с основной информацией по датасету
    """
    info = pd.DataFrame(index = df.columns)

    # Шаблон для создания колонок для info
    template = np.ones(len(df.columns))

    info['dtype'] = df.dtypes


    # Ищем часто встречаемый элемент
    vc_max_el = pd.Series(template, index=df.columns)
    vc_max_freq = pd.Series(template, index=df.columns)
    for col in df.columns:
        # Создаём вспомогательную переменную с подсчитыными элементами
        tmp  = df[col].value_counts(sort=True)
        # Длина столбца без NaN'ов
        len_without_nans = len(df) - len(df[df[col].isna()])
        vc_max_el[col] = tmp.idxmax()
        vc_max_freq[col] = round(
                            tmp.max() / len_without_nans,
                            3)

    info['vc_max_el'] = vc_max_el
    info['vc_max_freq'] = vc_max_freq

    # Считаем уникальные элементы в колонке
    nuniq = pd.Series(template, index=df.columns)
    for col in df.columns:
        nuniq[col] = df[col].nunique(dropna=False)
    # Приводим к удобному виду колонку
    nuniq = nuniq.astype('int64')
    info['nunique'] = nuniq

    # Cчитаем долю NaN. Аналогично делаем для zero и empty_str
    proportion_nans = pd.Series(template, index=df.columns)
    for col in df.columns:
        prop = df[col].isna().sum() / len(df)
        prop = round(prop, 3)
        # Для красоты нулевую долю обозначим как -1
        if prop != 0:
            proportion_nans[col] = prop
        else:
            proportion_nans[col] = '-1'

    info['nan'] = proportion_nans


    proportion_zeros = pd.Series(template, index=df.columns)
    for col in df.columns:
        prop = df[df[col] == 0][col].count() / len(df)
        prop = round(prop, 3)
        if prop != 0:
            proportion_zeros[col] = prop
        else:
            proportion_zeros[col] = '-1'

    info['zero'] = proportion_zeros


    proportion_empty = pd.Series(template, index=df.columns)
    for col in df.columns:
        prop = df[df[col] == ''][col].count() / len(df)
        prop = round(prop, 3)
        if prop != 0:
            proportion_empty[col] = prop
        else:
            proportion_empty[col] = '-1'

    info['empty_str'] = proportion_empty

    # Считаем trash_score
    info['trash_score'] = (
            # т.к до этого нулевые значения мы заменили на -1, на время вернём как было
            info['nan'].where(info['nan'] != '-1', 0) + (
            info['empty_str'].where(info['empty_str'] != '-1', 0)) + (
            info['zero'].where(info['zero'] != '-1', 0))
        )

    # Заменяем значения, которые меньше нашего vc_max_freq (если тот достиг порога)
    info['trash_score'].where(info['vc_max_freq'] < thr, info['vc_max_freq'], inplace=True)
    # Сортируем всю таблицу по значению trash_score
    info.sort_values('trash_score', ascending=False, inplace=True)
    # Возращаем нулевые значения в "-1"
    info['trash_score'].where(info['trash_score'] != 0, '-1', inplace=True)

    return info

def smart_corr(data: pd.DataFrame, hue: str, interval_cols: list=[], head: int=10, hm: bool=True): 
    '''
    Функция, которая возвращает коэффициент корреляции phik_k между признаками и hue

    Args:
    data - таблица, из которой мы берём данные
    hue - целевая переменная
    interval_cols - список колонок, которые принимают интервальные значения. Нужно для корректной работы phik
    hm - показывать ли heatmap
    head - показывает Top of head признаков

    Returns:
    pd.DataFrame первые 10 признаков, которые хорошо коррелированы с hue
    '''
    phik_matrix_corr = phik_matrix(data, interval_cols=interval_cols)
    if hm:
        heatmap(phik_matrix_corr)
    return phik_matrix_corr[hue].sort_values(ascending=False).to_frame().head(head)

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = tm()
        result = func(*args, **kwargs)
        end_time = tm() - start_time
        if isinstance(result, tuple):
            return *result, end_time
        return result, end_time
    return wrapper



'''
Функции ниже должны
    принимать:
        data, params (остальное пихайте через partial извне)
    возвращать:
        mapper (если есть. если нет, возвращаем None)
            объект-обученный-reductor с методом transform (нужен только для того чтобы вернуть его пользователю для работы с тестом)
        embedding
            2D / 3D embedding для отрисовки. будем также возвращать пользователю, если попросит
'''

# при желании, UMAP можно изменить, чтобы он тоже принимал предпосчитанные affinities, init
@timer
def make_umap(data, params, y=None):
    '''
    можно вшить y через partial для [semi-]supervised learning
    '''
    mapper = UMAP(**params).fit(data, y)
    return mapper, mapper.embedding_


@timer
def make_tsne(data, params, init=None, affinities=None):
    '''
    можно вшить init, affinities через partial, чтобы не считать по сто раз,
        если вы не хотите их менять
    '''
    rescaled_init = None
    if init is not None:
        rescaled_init = openTSNE.initialization.rescale(init, inplace=False, target_std=0.0001)
        
    # mapper_embedding - объект класса TSNEEmbedding - и маппер, и эмбеддинг в одном :)
    mapper_embedding = openTSNE.TSNE(**params).fit(data, initialization=rescaled_init, affinities=affinities)
    return None, mapper_embedding

@timer
def make_pca(data, params):
    mapper = PCA(**params).fit(data)
    embedding = mapper.transform(data)
    return mapper, embedding


def plot_dim_reduction(data, mapper_dict, default_features=None, default_hue_info=None, 
                       row_width=950, row_height=500, plotly_marker_size=1.5, bokeh_marker_size=3, return_results=False):
    '''
    Функция принимает на вход данные и набор 2D/3D dimension-редукторов через mapper_dict.
    Отрисовывает эмбеддинги этих данных в наиболее удобных форматах: 3D - plotly, 2D - bokeh с CDS sharing'ом
    
    
    data - pd.DataFrame со всеми необходимыми данными - hue_cols, features
    
    mapper_dict - словарь специального вида (см. ниже)
    
    default_features: array of strings - фичи которые будут использоваться для вычисления функции расстояния,
        если для reductor`а не указано иного
        
    default_hue_info: namedtuple - вида (hue-колонка-строка, is_categorical),
        инфа о hue-колонке, которая будет использоваться, если для reductor`а не указано иного
    
    row_width: int - ширина ряда из картинок
        узнать - рисуйте пустую bokeh.plotting.figure, увеличивая width,
        пока фигура не станет занимать все свободное место в ширину
        
    row_height: int
        желаемая высота ряда
        
    .._marker_size: размер точек на plotly и bokeh графиках
    
    return_results: bool - возвращать ли словарь {mapper_name: {'mapper': mapper, 'embedding': embedding}, ...}
    
    returns
        results: dict if return_results
        
        Note: для t-SNE mapper=embedding и лежит по ключу 'embedding'!
            Это объект класса TSNEEmbedding, это "обертка" над эмбеддингом.
            У него есть метод transform, а также его можно воспринимать как эмбеддинг и, например, слайсить и рисовать

    Cпециальный вид подаваеммого словаря:
    mapper_dict = {
        'tsne perplex=30 exagger=4 dof=0.5 metric=l2': {
            'params': {
                'n_jobs': 1,
                'perplexity': 30,
                'verbose': False,
                'n_components': 2,
                'early_exaggeration_iter': 300,
                'early_exaggeration': 24,
                'n_iter': 1000,
                'exaggeration': 4,
                'metric': 'l2',
                'dof': 0.5,
                'neighbors': 'pynndescent'
            },
            'func': make_tsne_preset_l2,
        },
        'tsne perplex=30 exagger=4 dof=0.5 metric=cosine': {
            'params': {
                'n_jobs': 1,
                'perplexity': 30,
                'verbose': False,
                'n_components': 2,
                'exaggeration': 4,
                'metric': 'cosine',
                'early_exaggeration_iter': 300,
                'early_exaggeration': 24,
                'n_iter': 1000,
                'dof': 0.5,
                'neighbors': 'pynndescent'
            },
            'func': make_tsne_preset_cos,
        },
        'UMAP 2D n_neighbors=100 metric=l2': {
            'params': {
               'n_neighbors': 100,
               'min_dist': 0.1,
               'metric': 'l2',
               'n_jobs': 1,
               'verbose': False,
               'n_components': 2
            },
            'func': make_umap,
        }
        }
    '''
    if default_hue_info is None:
        default_hue_info = None, None
    
    output_notebook() # bokeh render in notebook
    bokeh_first_time = True
    
    plotly_figs, bokeh_figs = [], []
    results = dict()
    for mapper_name in tqdm(mapper_dict):
        mapper_props = mapper_dict[mapper_name]
        params, features = mapper_props['params'], mapper_props.get('features', default_features)
        if features is None:
            raise ValueError(f'Мапперу {mapper_name} нужно указать фичи')
            
        mapper, embedding, time_passed = mapper_props['func'](data[features].values, params)
        results[mapper_name] = {
            'embedding': embedding,
            'mapper': mapper
        }
        
        # СБОР ИНФОРМАЦИИ ДЛЯ ОТРИСОВКИ
        
        x, y = embedding[:, 0], embedding[:, 1]
        hue_info = mapper_props.get('hue', default_hue_info)
        hue_field_name, hue_is_categorical = hue_info if hue_info is not None else (None, None)
        
        if embedding.shape[1] == 3: # plotly 3D render
            z = embedding[:, 2]
            plot_data = {'x': x, 'y': y, 'z': z}
            if hue_field_name is not None:
                if hue_is_categorical:
                    # простой способ показывать легенду вместо colorbar
                    plot_data[hue_field_name] = data[hue_field_name].astype(str)
                else:
                    # в этом случае будет показываться colorbar
                    plot_data[hue_field_name] = data[hue_field_name]
            
            plotly_fig = px.scatter_3d(plot_data, x='x', y='y', z='z', title=mapper_name, color=hue_field_name)
            plotly_figs.append(plotly_fig)
            
        else: # bokeh render with CDS sharing
            if bokeh_first_time:
                source = ColumnDataSource(data)
                bokeh_first_time = False
                
            x_name = f'{mapper_name}_x'
            y_name = f'{mapper_name}_y'
            source.data[x_name] = x
            source.data[y_name] = y
            
            # набор инструментов
            # можете добавить еще какие хотите
            bokeh_fig = figure(title=mapper_name, tools=['pan', 'wheel_zoom', 'box_select', 'lasso_select', 'reset', 'box_zoom'])
            
            if hue_is_categorical is None: # если не во что красить
                bokeh_fig.scatter(x=x_name, y=y_name, size=bokeh_marker_size, source=source)
                
            elif hue_is_categorical: # Если hue категориальный, у нас будет легенда с возможностью спрятать отдельные hue
                # scatter -> label_name требует строку. Поэтому делаем из числовых категорий строки
                # Сортируем числа, потом делаем строки для корректной сортировки
                uniques = np.sort(data[hue_field_name].unique()).astype(str)
                
                # Настраиваем палитры
                n_unique = uniques.shape[0]
                if n_unique == 2:
                    palette = tol['Bright'][3][:2]
                elif n_unique == 3:
                    palette = tol['HighContrast'][3]
                elif n_unique in tol['Bright']:
                    palette = tol['Bright'][n_unique]
                else:
                    palette = magma(n_unique)
                
                # Делаем через for чтобы поддерживать legend.click_policy = 'hide'
                for i, hue_val in enumerate(uniques):
                    # Будем рисовать только ту дату, где hue_col == hue_val
                    condition = (data[hue_field_name].astype(str) == hue_val).tolist()
                    view = CDSView(filter=BooleanFilter(condition))
                    
                    # Рисуем эмбеддинги
                    bokeh_fig.scatter(x=x_name, y=y_name, size=bokeh_marker_size,
                                      source=source, view=view, legend_label=hue_val, color=palette[i])
                
                # Добавляем легенде возможность спрятать по клику
                bokeh_fig.legend.click_policy = 'hide'
                
            else: # Если hue числовой, у нас будет colorbar
                # Настраиваем цветовую палитру
                min_val, max_val = data[hue_field_name].min(), data[hue_field_name].max()
                color = linear_cmap(
                    field_name=hue_field_name,
                    palette=magma(data[hue_field_name].nunique()),
                    low=min_val,
                    high=max_val
                )
                
                # Рисуем эмбеддинги
                plot = bokeh_fig.scatter(x=x_name, y=y_name, size=bokeh_marker_size, source=source, color=color)
                
                # Чуть настроим colorbar
                ticks = np.linspace(min_val, max_val, 5).round()
                ticker = FixedTicker(ticks=ticks)
                colorbar = plot.construct_color_bar(title=hue_field_name, title_text_font_size='20px', title_text_align='center',
                                                    ticker=ticker, major_label_text_font_size='15px')
                bokeh_fig.add_layout(colorbar, 'below')
            
            bokeh_fig.title.align = 'center'
            bokeh_figs.append(bokeh_fig)
    
    
    # ОТРИСОВКА
    # имеем запиханные в списки bokeh_figs и plotly_figs фигуры
    
    n_bokeh = len(bokeh_figs)
    if n_bokeh > 0:
        plot_width = round(row_width / (n_bokeh + 0.1))
        grid = gridplot([bokeh_figs], width=plot_width, height=row_height)
        show(grid)
    
    n_plotly = len(plotly_figs)
    if n_plotly > 0:
        plot_width = round(row_width / (n_plotly + 0.05))
        
        # plotly удобнее всего запихнуть в строку с помощью ipywidgets.widgets.HBox
        # его нужно вернуть
        plotly_widgets = []
        for i in range(n_plotly):
            fig = plotly_figs[i]
            layout = fig.layout
            layout.update({'width': plot_width, 'height': row_height,
                           'title_x': 0.5, 'title_font_size': 13, 'legend_itemsizing': 'constant'})
            fig.update_layout(legend= {})
            new_fig = go.FigureWidget(fig.data, layout=layout)
            new_fig.update_traces(marker_size=plotly_marker_size)
            plotly_widgets.append(new_fig)
            
        display(widgets.HBox(plotly_widgets))
        
    if return_results:
        return results
    

def print_diff_between_sets(left, right, feature, name_left, name_right):
      '''
      Выводит на экран количество уникальных значений признака feature в left и right,
      а также количество пересечений уникальных значений 
      '''
      print(f'Количество уникальных {feature} в {name_left}:', len(left[feature].unique()))
      print(f'Количество уникальных {feature} в {name_right}:', len(right[feature].unique()))
      intersect = np.intersect1d(right[feature].unique(), left[feature].unique())
      print(f'Количество совпадающих {feature}:', intersect.shape[0])
      print(f'Количество уникальных {feature} в {name_right}, которых нет в {name_left}:',
            len(np.setdiff1d(right[feature].unique(), intersect)))
      


# def mlflow_logging(train, models, exp, i):
#     with mlflow.start_run(run_name=f'Run {i}', experiment_id=exp.experiment_id) as run:
#         mlflow.log_param('shape of train dataset', train.shape)
#         for
#         mlflow.log_params(gbm.params)
#         mlflow.log_param('model_type', 'lgbm')
#         mlflow.log_metric('num early stoping', gbm.best_iteration)
#         mlflow.log_metric('roc auc score', gbm.best_score['valid_0']['auc'])

#         cols_pred = []
#         for col in train.columns:
#             if col != 'target':
#                 cols_pred.append(col)
                
#         importance = pd.DataFrame({
#             "feature" : cols_pred,
#             'importance' : gbm.feature_importance()
#         })
#         importance.sort_values('importance', ascending=False, inplace=True)
#         importance.to_csv(f'importance of {exp.name} {i}.csv', index=False)
        
#         #mlflow.log_artifact('feature_importance_the initial experiment.jpg')
#         mlflow.log_artifact(f'importance of {exp.name} {i}.csv')
#         mlflow.lightgbm.log_model(gbm, 'lgbm')