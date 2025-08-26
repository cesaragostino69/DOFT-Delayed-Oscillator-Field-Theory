Guía de Ejecución: De Cero a Simulación
Este documento describe el proceso completo para configurar el entorno de simulación y ejecutar los experimentos de la Fase 1 desde una máquina limpia.

Paso 1: Configuración del Entorno

Estos pasos solo necesitan realizarse una vez por cada máquina.

1.1. Clonar el Repositorio

Primero, descarga el código fuente desde el repositorio de GitHub.

git clone https://github.com/cesaragostino/DOFT-Delayed-Oscillator-Field-Theory.git
cd DOFT-Delayed-Oscillator-Field-Theory

1.2. Crear el Entorno de Conda

El proyecto utiliza un entorno Conda para gestionar todas las dependencias de Python. El archivo environment.yml en la raíz del proyecto define todo lo necesario.

# Este comando leerá el archivo y creará un entorno llamado 'doft_v12'
conda env create -f environment.yml

1.3. Activar el Entorno

Una vez creado, activa el entorno para poder usar las herramientas instaladas.

conda activate doft_v12

Nota: Deberás ejecutar este comando cada vez que abras una nueva terminal para trabajar en el proyecto.

Paso 2: Ejecución de los Experimentos de Fase 1

Con el entorno configurado y activado, ya puedes lanzar las simulaciones.

2.1. El Script Principal

El script scripts/run_phase1.sh es el único punto de entrada que necesitas. Se encarga de configurar las variables, verificar el entorno y lanzar el simulador de Python con la configuración correcta de la Fase 1.

2.2. Ejecución en CPU

Para ejecutar todas las simulaciones de la Fase 1 utilizando los núcleos de la CPU:

# La variable de entorno USE_GPU=0 le indica al script que no busque una GPU.
# N_JOBS=4 usará 4 procesos en paralelo. Ajústalo al número de cores de tu máquina.
USE_GPU=0 N_JOBS=4 bash scripts/run_phase1.sh

2.3. Ejecución en GPU

Si tu máquina tiene una GPU NVIDIA compatible con CUDA y has instalado los wheels de PyTorch, puedes acelerar la simulación de la siguiente manera:

# USE_GPU=1 activará el uso de la GPU.
# En modo GPU, N_JOBS generalmente se mantiene en 1.
USE_GPU=1 N_JOBS=1 bash scripts/run_phase1.sh

El script verificará si la GPU está disponible. Si no lo está, te avisará y continuará la ejecución en la CPU.

Paso 3: Resultados

Al finalizar la ejecución, el script creará un nuevo directorio dentro de la carpeta /results. El nombre del directorio incluirá la fecha y hora de la corrida, por ejemplo: results/phase1_run_20250825_183000.

Dentro de esa carpeta encontrarás los artefactos de la simulación (runs.csv, blocks.csv, etc.), listos para ser analizados.

