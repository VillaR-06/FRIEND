import os
import discord
from discord.ext import commands
import uuid
import numpy as np
from keras.models import load_model
from keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D
from PIL import Image, ImageOps
import io

####LISTA DE COMANDOS####
###
##
#

Comandos = """
Aqui tienes los comandos con los que te puedo ayudar.

/higiene = Puedes conocer consejos sobre la higiene de tu perrito.
/ejercicio = Me encantara comentarte algunos ejercicios recomendados para tu canino.
/raza = Mira información sobre todas las razas que tengo disponible.
/veterinaria = Te mostrare las ciudades en las que tengo conocimiento de la existencia de veterinarias.
/amigo = Muestrame una imagen png o jpg de tu canino, te ayudare a identificar su raza, por medio de sus caracteristicas.
"""

####LISTA DE RAZAS####
###
##
#

Razas = """
Aqui tienes la lista de razas de las que tengo conocimientos sobre su cantidad de ejercicio y comida ideal.

/pastoraleman
/goldenretriever
/beagle
/chihuahua
/rottweiller
"""

####VETERINARIAS####
###
##
#

Veterinarias = """
Guatemala:

Ciudad de Guatemala - /ciudadguatemala
Mixco - /mixco
Villa Nueva - /villanueva
"""

####DIRECCIONES Y DATOS####
###
##
#

###GUATEMALA###
##
#

Ciudad_de_Guatemala = """
Aquí tienes una lista de veterinarias en Ciudad de Guatemala, incluyendo direcciones y números de contacto:

*Hospital Veterinario de la Facultad de Medicina Veterinaria*
Dirección: Edificio M8, Ciudad Universitaria, Zona 12, Ciudad de Guatemala
Teléfono: 2418-8000 Ext. 1592-1593
Horario: Lunes a viernes, 8:00 a 13:00 horas.

*Hospital y Clínicas Veterinarias Dana*
Dirección: Cardales de Cayalá, Ciudad de Guatemala
Teléfono: 2231-9900
Horario: Emergencias 24/7.

*Hospital Veterinario Palvet*
Dirección: 18 Avenida 19-39, Zona 10, Ciudad de Guatemala
Teléfono: No especificado
Horario: Lunes a viernes, 8:00 a 17:00 horas; Sábados, 9:00 a 13:00 horas.

*Hospital de Mascotas*
Dirección: Cerca del Trébol, San Juan Roosevelt, Ciudad de Guatemala
Teléfono: 5518-8301
Horario: Atención 24/7.

*Animédica Hospital Veterinario*
Dirección: 10 Avenida 14-46, Zona 10, Ciudad de Guatemala
Teléfono: 2308-5792
Horario: Lunes a viernes, 8:00 a 17:00 horas; Sábados, 8:00 a 13:00 horas.

*PAWS Veterinaria (Sucursal Portal El Zapote)*
Dirección: Centro Comercial Portal El Zapote, Zona 2, Ciudad de Guatemala
Teléfono: No especificado
Servicios: Veterinario, grooming y petshop.

Esta información te permitirá contactar con varias opciones para atención veterinaria en la ciudad.
"""
Villa_Nueva = """
Aquí tienes una lista de veterinarias en Villa Nueva, Guatemala, con sus respectivas direcciones:

*PAWS Metrocentro*
Dirección: Centro Comercial Metrocentro, Villa Nueva
Servicios: Veterinaria, grooming, venta de alimentos y accesorios para mascotas.

*Veterinaria Caleb*
Dirección: Santa Clara, Villa Nueva
Teléfono: 50023054
Servicios: Atención veterinaria profesional.

*Veterinaria Los Carlos*
Dirección: 5 Av, Villa Nueva, Guatemala
Email: vetcarlos@hotmail.com
Servicios: Atención veterinaria general.

*Centro Veterinario La Villa*
Dirección: Colonia Enriqueta, 146, Zona 5, Villa Nueva, Guatemala
Teléfono: 66314600
Servicios: Veterinaria y atención de emergencias.

*Semivet*
Dirección: Calle Real 19-16, Cabecera Municipal Zona 10, San Miguel Petapa y Villa Nueva, Guatemala
Teléfono: +(502) 2306-4056
Servicios: Atención veterinaria y productos para mascotas.

*Clínica Veterinaria (Atención 24 horas)*
Dirección: 1a Avenida 1-1 Zona 1, Villa Nueva, Guatemala
Servicios: Abierto las 24 horas todos los días.

Cada una de estas veterinarias ofrece diversos servicios para el cuidado de tus mascotas.
"""

Mixco = """
Aquí tienes una lista de veterinarias en Mixco, Guatemala, junto con sus direcciones:

*Clínica Veterinaria 24 Horas*
Dirección: 12a Avenida 12-12, Zona 12, Mixco, Guatemala.
Atención: Abierto 24 horas, todos los días de la semana.

*Mater Et Magistra*
Dirección: Plaza Helios, Blvd. San Cristóbal Z.8, Mixco.
Teléfono: 2480-7505. 

*Purina*
Dirección: 14 Av. 16-70, Zona 4, Mixco (Cond. El Naranjo).
Teléfono: 2427-4000. 

*Veterinaria Mi Mascota*
Dirección: 9a Calle, La Ceiba, Zona 10, Mixco.
Información adicional: Acceso a información de tráfico en tiempo real disponible en Waze.

*Hospital Veterinario Pet Care*
Dirección: 3ra Calle 11-31, Colonia Nueva Monserrat, Zona 3, Mixco.
Teléfono: No especificado.

*Veterinaria Centro Comercial Montserrat*
Dirección: Calzada San Juan 14-25, Zona 4, Mixco.
Teléfono: 2431-4552.
Horario: Lunes a Domingo de 8:00 AM a 8:00 PM.

*Clínica Veterinaria Ixchel*
Dirección: 3 Calle 18-72, Centro Comercial San Cristóbal, Zona 8 de Mixco, Local 20.
Teléfono: No especificado.

*Hospital Veterinario San Ignacio*
Dirección: 12 Avenida, Zona 7, Mixco.
Teléfono: 2436-1154.
Horario: Domingo de 08:01 a 16:00; Lunes a Sábado de 07:30 a 17:00 (actualmente cerrado).

Espero que esta información te sea útil para encontrar la veterinaria que necesitas en Mixco.
"""

##CONSEJOS GENERALES

limpieza = """Aquí tienes consejos resumidos para la limpieza y cuidado de perros:
*Cuidado del pelaje:*
Cepillado Regular: Cepilla el pelaje antes del baño para eliminar nudos y pelo muerto.
Baño: Usa agua tibia y un champú suave. Enjuaga bien para evitar irritaciones.
Secado: Seca al perro con una toalla y, si es posible, usa un secador a baja temperatura.

*Limpieza General:*
Higiene Bucal: Limpia los dientes de tu perro regularmente para prevenir problemas dentales.
Limpieza de Oídos y Ojos: Limpia suavemente estas áreas con un paño húmedo.
Corte de Uñas: Mantén las uñas cortas para evitar molestias al caminar.

*Productos de Limpieza:*
Usa Productos Naturales: Opta por limpiadores no tóxicos y biodegradables para mantener la seguridad de tu mascota.
Desinfección de Artículos: Lava la cama, juguetes y platos del perro con agua y jabón, evitando productos químicos peligrosos.
Ventilación: Asegúrate de que el área esté bien ventilada al limpiar con productos químicos.
Estos consejos ayudarán a mantener a tu perro limpio y saludable, así como su entorno.
"""

Exercise = """
Aquí tienes una versión simplificada de los ejercicios ideales para perros:

*Ejercicios en Casa*:
Caminatas por el Pasillo: Pasea por la casa usando juguetes como motivación.
Juegos de Búsqueda: Esconde premios o juguetes y deja que tu perro los encuentre.
Circuitos de Objetos: Crea un recorrido con sillas o cajas para que tu perro salte o rodee.
Tira y Afloja: Juega con un juguete resistente para liberar energía.

*Ejercicios al Aire Libre*:
Caminatas o Trotar: Paseos regulares o trotar para ejercitarse juntos.
Parques para Perros: Socializa y juega en un parque especializado.
Natación: Excelente ejercicio de bajo impacto en climas cálidos.
Subir y Bajar Escaleras: Usa escaleras para un buen ejercicio físico.
Jugar a Atrapar: Lanza una pelota o frisbee para mantenerlo activo.
Estas actividades son divertidas y ayudan a mantener a tu perro saludable y feliz."""


####CONSEJOS POR RAZA####
###
##
#

Pastor_Aleman = """
El peso promedio de un pastor alemán varía en las diferentes etapas de su vida:

*Cachorros*

3 meses: Machos: 11.4 - 14.3 kg; Hembras: 8.6 - 11.9 kg.
6 meses: Machos: 21.2 - 26.7 kg; Hembras: 16 - 22.3 kg.
1 año: Machos: 29 - 37.8 kg; Hembras: 21.3 - 30.8 kg.

*Adultos*

Machos (a partir de 1 año): 30 - 40 kg.
Hembras (a partir de 1 año): 22 - 32 kg


*Alimentación*

Cachorros (8 semanas - 6 meses): 3-4 comidas al día, entre 1-2 tazas por comida, con un enfoque en piensos ricos en proteínas y grasas.

Adultos (6 meses - 7 años): 2-3 tazas de pienso de alta calidad al día, divididas en dos comidas. Debe contener al menos un 25 porciento de proteínas y un 15 porciento de grasas.

Anciano (más de 7 años): 1.5-2.5 tazas al día, dependiendo de su nivel de actividad.

*Ejercicio*
Los pastores alemanes requieren al menos 1-2 horas de ejercicio diario, que puede incluir caminatas, juegos y entrenamiento.
"""

Rottweiller = """
El peso promedio de un Rottweiler varía según su etapa de vida:

*Cachorro (0 a 12 meses)* 

2 a 3 meses: Machos: 17.1 a 19.8 kg, Hembras: 12.7 a 16.6 kg.
6 meses: Machos: 31.5 a 35.6 kg, Hembras: 23.8 a 30.9 kg.
1 año: Machos: 46.3 a 54.5 kg, Hembras: 33.2 a 44.8 kg.

*Adulto (1 a 7 años)*

Machos: 39 a 61 kg.
Hembras: 36 a 45 kg.

*3. Anciano (más de 7 años)*
Mantienen un peso similar al de la etapa adulta, aunque puede variar según la salud y el nivel de actividad.

Es importante monitorear el peso para prevenir problemas de salud relacionados con la obesidad.


*Alimentación*
Cachorro (2 a 12 meses)

2 a 3 meses: 4 raciones diarias de leche materna o fórmula.
4 a 5 meses: 3 raciones diarias de pienso específico para cachorros.
6 a 12 meses: 2 a 3 raciones diarias, aumentando gradualmente la cantidad hasta un máximo de 1,500 gramos diarios.

Adulto (1 a 7 años)

Cantidad diaria: Entre 1,200 y 2,000 gramos, dependiendo del peso y actividad. Se debe dividir en 2 a 3 raciones.

Anciano (más de 7 años)

Cantidad diaria: Aproximadamente 1,000 a 1,500 gramos, ajustando según el nivel de actividad y salud.

Es esencial consultar con un veterinario para ajustar la dieta según necesidades específicas y condiciones de salud.

*Ejercicio*

Se recomienda un mínimo de 30 a 40 minutos de ejercicio diario, que puede incluir caminatas, trotes y juegos.
"""

Beagle = """
El peso promedio de un Beagle varía según su etapa de vida:

Cachorro (hasta 12 meses)
- 2 meses: 2,7 - 4,4 kg[5]
- 3 meses: 3,9 - 4,7 kg (dependiendo del sexo)
- 6 meses: 7 - 8,5 kg (dependiendo del sexo)

Adulto (1 a 7 años)
- Hembras: 9 - 10 kg
- Machos: 10 - 11 kg
- En general: 10 - 13,5 kg

Anciano (7 años en adelante)
El peso ideal para un Beagle senior es similar al de un adulto, pero es importante mantener un control más estricto para evitar el sobrepeso, que es común en esta etapa.

Es importante notar que estos son promedios y el peso puede variar según el individuo y su nivel de actividad. Siempre es recomendable consultar con un veterinario para determinar el peso ideal de tu Beagle específico.

*Alimento*

Cachorro (2 a 6 meses)
2 a 4 meses: 70-90 gramos diarios, divididos en 3-4 comidas13.
4 a 6 meses: 90 gramos diarios, divididos en 3 comidas17.

Adulto (6 meses en adelante)
Adultos: 200-300 gramos diarios, divididos en 2 comidas48. 

La cantidad exacta depende del peso y nivel de actividad del perro.

*Ejercicio*
Los Beagles requieren al menos una hora de ejercicio diario, aunque lo ideal son dos horas o más. 
Este ejercicio puede incluir paseos, correr, y juegos como buscar objetos. Es recomendable hacer al menos tres paseos al día, asegurando que uno de ellos sea prolongado, de aproximadamente una hora y media
"""

Chihuahua = """
El peso promedio de un chihuahua varía en sus tres etapas de vida:

Cachorros (0-12 meses)
Al nacer: 80-119 g.
1 mes: 270-370 g.
3 meses: 625-855 g.
6 meses: 1.10-1.45 kg.

Adultos (1-8 años)
Machos: 1.4-3 kg.
Hembras: 1-2.7 kg.

Ancianos (8+ años)
El peso se mantiene similar al de los adultos, aunque puede variar ligeramente según la salud y actividad del perro.

*Alimento*
Para un chihuahua, la cantidad ideal de comida varía según su etapa de vida:

Cachorros (0-12 meses)
2 meses: 30-40 g por comida, 4 veces al día.
3 meses: 40-50 g por comida, 3 veces al día.
4-6 meses: 31-120 g diarios.
6-10 meses: 27-105 g diarios.

Adultos (1-8 años)
1 kg: 24-45 g.
2 kg: 41-75 g.
3 kg: 55-101 g.
Generalmente, entre 50 y 150 g al día, repartidos en 2 o 3 comidas.

Ancianos (8+ años)
1 kg: 25-28 g.
2 kg: 43-48 g.
3 kg: 58-64 g.
La cantidad puede ser menor y repartida en más comidas

*Ejercicio*

Los chihuahuas necesitan entre 20 y 60 minutos de ejercicio diario. 
Para los adultos, se recomienda aproximadamente 30 minutos, que pueden incluir paseos y juegos. Los cachorros requieren sesiones más cortas pero más frecuentes, mientras que los perros mayores pueden beneficiarse de paseos suaves y más cortos para evitar el sobreesfuerzo. 
Es esencial adaptar la actividad a su nivel de energía y salud, asegurando que no se cansen en exceso.
"""

Golden_Retriever = """
El peso promedio de un Golden Retriever en sus tres etapas de vida es el siguiente:

Cachorros (0-12 meses)
3 meses: Machos 9-12 kg, Hembras 8-11 kg.
6 meses: Machos 21-23 kg, Hembras 18-22 kg.
1 año: Machos 29-33 kg, Hembras 25-30 kg.

Adultos (1-7 años)
Machos: 29-34 kg.
Hembras: 25-32 kg.

Seniors (7+ años)
El peso puede mantenerse similar al de los adultos, pero puede variar según la salud y actividad del perro.

*Alimento*

Para un Golden Retriever, la cantidad de comida en gramos varía según su etapa de vida:

Cachorros (0-12 meses)
2-4 meses: 3-4 comidas al día, aproximadamente 100-200 g por comida.
4-6 meses: 3 comidas al día, aproximadamente 250 g por comida.
6-12 meses: 2 comidas al día, aproximadamente 400-600 g en total.

Adultos (1-7 años)
Cantidad diaria: Entre 400 y 600 g de comida seca, repartidos en dos comidas.

Seniors (7+ años)
Cantidad diaria: Similar a los adultos, pero puede ajustarse a menos según su actividad y salud.

*Ejercicio*

Cachorros (0-12 meses)
Ejercicio diario: 5 minutos por mes de edad, dos veces al día.

Adultos (1-7 años)
Ejercicio diario: 1 a 2 horas de actividad física moderada a intensa.

Seniors (7+ años)
Ejercicio diario: 30 a 60 minutos, ajustando la intensidad según su salud.
Este enfoque asegura que tu Golden Retriever se mantenga saludable y feliz en cada etapa de su vida.
"""

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)
client = discord.Client(intents=intents) 
# Cargar modelo de IA
class CustomDepthwiseConv2D(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

####Cargar modelo de IA####
###
##
#

custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
model = load_model("keras_model.h5", custom_objects=custom_objects, compile=False)
class_names = open("labels.txt", "r").readlines()


np.set_printoptions(suppress=True)

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

@bot.event
async def on_ready():
    print(f'Hemos iniciado sesión como {bot.user}')

@bot.command()
async def hola(ctx):
    await ctx.send("Me presento, soy FRIEND, el bot amigo que te ayudara con el cuidado de tu canino, para empezar di /comandos")

@bot.command()
async def higiene(ctx):
    await ctx.send(limpieza)

@bot.command()
async def comandos(ctx):
    await ctx.send(Comandos)

@bot.command()
async def raza(ctx):
    await ctx.send(Razas)

@bot.command()
async def veterinarias(ctx):
    await ctx.send(Veterinarias)

@bot.command()
async def ciudadguatemala(ctx):
    await ctx.send(Ciudad_de_Guatemala)

@bot.command()
async def mixco(ctx):
    await ctx.send(Mixco)

@bot.command()
async def villanueva(ctx):
    await ctx.send(Villa_Nueva)

@bot.command()
async def ejercicio(ctx):
    await ctx.send(Exercise)

@bot.command()
async def pastoraleman(ctx):
    await ctx.send(Pastor_Aleman)

@bot.command()
async def rottweiller(ctx):
    await ctx.send(Rottweiller)

@bot.command()
async def beagle(ctx):
    await ctx.send(Beagle)

@bot.command()
async def chihuahua(ctx):
    await ctx.send(Chihuahua)

@bot.command()
async def goldenretriever(ctx):
    await ctx.send(Golden_Retriever)


@bot.command()
async def amigo(ctx):
    if len(ctx.message.attachments) == 0:
        await ctx.send("No se ha adjuntado ninguna imagen. Por favor, sube una imagen junto con el comando.")
        return

    attachment = ctx.message.attachments[0]
    if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        await ctx.send("El archivo adjunto no es una imagen válida. Por favor, sube una imagen en formato PNG, JPG, JPEG o GIF.")
        return

    unique_filename = f"{uuid.uuid4()}{os.path.splitext(attachment.filename)[1]}"
    save_path = os.path.join("imagenes", unique_filename)

    await attachment.save(save_path)
    await ctx.send(f"Imagen guardada exitosamente como {unique_filename}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg']):
                try:
                    image_bytes = await attachment.read()
                    class_name, confidence_score = predict_image(image_bytes)
                    await message.channel.send(f"Clase: {class_name}\n*Confianza:* {confidence_score:.2f}")
                except Exception as e:
                    await message.channel.send(f"Hubo un error al procesar la imagen: {e}")
                return  # Solo procesar una imagen a la vez

    # Procesar comandos existentes
    await bot.process_commands(message)

bot.run()
client.run()