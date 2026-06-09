# TalentBlik

## Descripción

TalentBlik es una plataforma de football intelligence y scouting desarrollada para apoyar el análisis, seguimiento y evaluación de jugadores daneses elegibles para la selección nacional.

La herramienta integra datos de Wyscout (Opta) y FotMob para construir perfiles de jugador, sistemas de role-fit, radares específicos por rol, búsqueda de perfiles similares y simulación de convocatorias mediante un Shadow Squad interactivo.

---

## Objetivo

El objetivo del proyecto es proporcionar una herramienta de apoyo a la toma de decisiones que permita:

* Identificar talento elegible para la selección nacional.
* Evaluar jugadores según funciones tácticas específicas.
* Comparar perfiles similares.
* Analizar el encaje de un jugador dentro de un modelo de juego concreto.
* Construir convocatorias y plantillas de trabajo de forma visual.

---

## Fuentes de datos

### Wyscout (Opta)

Utilizado para:

* Role Fit
* Role Radar
* Similar Players
* Tactical Fit
* Shadow Squad

Se emplean métricas avanzadas de rendimiento por jugador para construir perfiles funcionales y arquetipos.

### FotMob

Utilizado como fuente complementaria para:

* Estadísticas de temporada
* Información contextual
* Visualización de datos de competición

---

## Metodología

### Pipeline de datos

Los datos son procesados mediante notebooks de Jupyter para:

* Limpieza de datos
* Normalización de competiciones
* Unificación de jugadores
* Filtrado de jugadores daneses
* Integración Wyscout + FotMob

El resultado final se almacena en un dataset parquet optimizado para la aplicación.

### Framework de Roles

La clasificación de jugadores se basa en un sistema propio de arquetipos futbolísticos definido en:

`player_role_templates_v30.xlsx`

Ejemplos:

* Proactive Goalkeeper
* Reactive Goalkeeper
* Mobile + Initiative Centre Back
* Physical + Safe Centre Back
* Creative Fullback
* Powerhouse Wingback
* Passing Progressor
* Deep Runner
* Wide Dribbler
* False Nine
* Target Man
* Complete Forward

### Role Fit

El sistema evalúa el encaje de cada jugador respecto a distintos roles mediante:

* Métricas Wyscout
* Pesos específicos por rol
* Referencias de rendimiento
* Ajustes positivos y negativos mediante direction

### Similar Players

La búsqueda de jugadores similares utiliza:

* Similaridad de perfil estadístico
* Nivel competitivo
* Compatibilidad posicional

Con el objetivo de identificar alternativas funcionalmente parecidas.

### Shadow Squad

Permite construir una convocatoria interactiva en una estructura táctica 4-2-3-1.

Cada posición define:

* Roles prioritarios
* Nivel de encaje
* Necesidades tácticas del modelo de juego

---

## Estructura del proyecto

```text
TalentBlik/
├── App1g.py
├── README.md
├── requirements.txt
└── data/
    ├── denmark_player_pool_union.parquet
    ├── player_role_templates_v30.xlsx
    └── wyscout_metrics_categories_updated.xlsx
```

---

## Instalación

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar aplicación:

```bash
python App1g.py
```

---

## Tecnologías utilizadas

* Python
* Dash
* Plotly
* Pandas
* NumPy
* PyArrow
* OpenPyXL

---

## Autor

Santiago Bahamón

Trabajo Fin de Máster (TFM)

Football Intelligence & Scouting Analytics
