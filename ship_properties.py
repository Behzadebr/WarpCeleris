class ShipProperties:
    def __init__(self):
        self.ship_posx: float = 2300.0  # initial ship position, if initially inside domain, the initial free surface must include the ship displacement 
        self.ship_posy: float = 500.0
        self.ship_width: float = 10.0  # ship beam
        self.ship_length: float = 30.0  # ship length
        self.ship_draft: float = 2.0  # max draft in m
        self.ship_heading: float = 0.0  # 0=moving to the east
        self.ship_dx: float = 0.0  # incremental motion of vessel in x-direction during a time step
        self.ship_dy: float = 0.0  # incremental motion of vessel in y-direction during a time step
        self.ship_c1a: float = 0.0  # various coefficients which prescribe the ship pressure distribution, following eqn 53 in Aykut & Lynett, 2021, pre-calcing coefs here
        self.ship_c1b: float = 0.0
        self.ship_c2: float = 0.0
        self.ship_c3a: float = 0.0
        self.ship_c3b: float = 0.0
        self.ship_velx: float = 0.0
        self.ship_vely: float = 0.0
        self.ship_speed: float = 0.0
        self.last_dx: float = 0.0  # Last horizontal movement
        self.last_dy: float = 0.0  # Last vertical movement
