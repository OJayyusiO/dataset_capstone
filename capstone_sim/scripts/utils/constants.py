"""Shared constants for the CARLA dataset capture and evaluation pipeline."""

CLASS_NAMES = {
    0: 'car',
    1: 'ambulance',
    2: 'bus',
    3: 'truck',
    4: 'police_car',
    5: 'fire_truck',
    6: 'bike',
}

CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

BLUEPRINT_TO_CLASS = {
    # Cars (class 0)
    'vehicle.tesla.model3': 0,
    'vehicle.audi.tt': 0,
    'vehicle.audi.a2': 0,
    'vehicle.audi.etron': 0,
    'vehicle.bmw.grandtourer': 0,
    'vehicle.chevrolet.impala': 0,
    'vehicle.citroen.c3': 0,
    'vehicle.dodge.charger_2020': 0,
    'vehicle.ford.mustang': 0,
    'vehicle.jeep.wrangler_rubicon': 0,
    'vehicle.lincoln.mkz_2017': 0,
    'vehicle.mercedes.coupe': 0,
    'vehicle.micro.microlino': 0,
    'vehicle.mini.cooper_s': 0,
    'vehicle.nissan.micra': 0,
    'vehicle.nissan.patrol': 0,
    'vehicle.seat.leon': 0,
    'vehicle.toyota.prius': 0,
    # Ambulance (class 1)
    'vehicle.ford.ambulance': 1,
    # Bus (class 2)
    'vehicle.mitsubishi.fusorosa': 2,
    # Truck (class 3)
    'vehicle.carlamotors.carlacola': 3,
    'vehicle.tesla.cybertruck': 3,
    # Police car (class 4)
    'vehicle.dodge.charger_police': 4,
    'vehicle.dodge.charger_police_2020': 4,
    # Fire truck (class 5)
    'vehicle.carlamotors.firetruck': 5,
    # Bikes (class 6)
    'vehicle.harley-davidson.low_rider': 6,
    'vehicle.kawasaki.ninja': 6,
    'vehicle.vespa.zx125': 6,
    'vehicle.yamaha.yzf': 6,
    'vehicle.bh.crossbike': 6,
    'vehicle.diamondback.century': 6,
    'vehicle.gazelle.omafiets': 6,
}

# Filtering thresholds for bounding boxes
MAX_DISTANCE = 80.0
MIN_BBOX_AREA = 200
MIN_BBOX_SIDE = 10
MIN_VISIBILITY = 0.15

# Colors (BGR) for video annotation
CLASS_COLORS = {
    0: (0, 255, 0),      # car - green
    1: (0, 0, 255),      # ambulance - red
    2: (255, 165, 0),    # bus - orange
    3: (255, 0, 255),    # truck - magenta
    4: (255, 255, 0),    # police_car - cyan
    5: (0, 69, 255),     # fire_truck - orange-red
    6: (147, 20, 255),   # bike - pink
}

# Colors for matplotlib charts
CLASS_COLORS_PLT = {
    0: '#2ecc71', 1: '#e74c3c', 2: '#f39c12', 3: '#9b59b6',
    4: '#00bcd4', 5: '#ff5722', 6: '#e91e63',
}
