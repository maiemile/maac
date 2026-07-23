"""
  RE problem adaptations to DESDEO by @maiemile, based on the 'reproblems' package.

  A real-world multi-objective problem suite (the RE benchmark set) 
  Reference:
  Ryoji Tanabe, Hisao Ishibuchi, "An Easy-to-use Real-world Multi-objective Problem Suite" Applied Soft Computing. 89: 106078 (2020)

  https://github.com/ryojitanabe/reproblems/blob/master/reproblem_python_ver/reproblem.py
"""

from desdeo.problem import Problem
from desdeo.problem.schema import (
    Objective,
    ObjectiveTypeEnum,
    Problem,
    Variable,
    VariableTypeEnum,
)


def re31() -> Problem:
    '''
    Two bar truss design.
    '''
    x_1 = Variable(
        name="x_1",
        symbol="x_1",
        variable_type=VariableTypeEnum.real,
        lowerbound=0.00001,
        upperbound=100.0,
    )
    x_2 = Variable(
            name="x_2",
            symbol="x_2",
            variable_type=VariableTypeEnum.real,
            lowerbound=0.00001,
            upperbound=100.0,
    )
    x_3 = Variable(
            name="x_3",
            symbol="x_3",
            variable_type=VariableTypeEnum.real,
            lowerbound=1.0,
            upperbound=3.0,
    )

    f_1 = Objective(
        name="f_1",
        symbol="f_1",
        func="x_1 * Sqrt(16.0 + (x_3 * x_3)) + x_2 * Sqrt(1.0 + x_3 * x_3)",
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=False,
        is_convex=False,
        is_twice_differentiable=True,
    )
    f_2 = Objective(
            name="f_2",
            symbol="f_2",
            func="(20.0 * Sqrt(16.0 + (x_3 * x_3))) / (x_1 * x_3)",
            objective_type=ObjectiveTypeEnum.analytical,
            is_linear=False,
            is_convex=False,
            is_twice_differentiable=True,
    )
    f_3 = Objective(
                name="f_3",
                symbol="f_3",
                func="" \
                "Max(x_1 * Sqrt(16.0 + (x_3 * x_3)) + x_2 * Sqrt(1.0 + x_3 * x_3) - 0.1, 0)" \
                "+ Max((20.0 * Sqrt(16.0 + (x_3 * x_3))) / (x_1 * x_3) - 100000.0, 0)"
                "+ Max(((80.0 * Sqrt(1.0 + x_3 * x_3)) / (x_3 * x_2)) - 100000.0, 0)",
                objective_type=ObjectiveTypeEnum.analytical,
                is_linear=False,
                is_convex=False,
                is_twice_differentiable=True,
        )

    return Problem(
        name="Two bar truss design",
        description="The two bar truss design problem",
        variables=[x_1, x_2, x_3],
        objectives=[f_1, f_2, f_3],
    )


def re32() -> Problem:
    '''
    Welded beam design.
    '''

    x_1 = Variable(
        name="x_1",
        symbol="x_1",
        variable_type=VariableTypeEnum.real,
        lowerbound=0.125,
        upperbound=5.0,
    )
    x_2 = Variable(
        name="x_2",
        symbol="x_2",
        variable_type=VariableTypeEnum.real,
        lowerbound=0.1,
        upperbound=10.0,
    )
    x_3 = Variable(
        name="x_3",
        symbol="x_3",
        variable_type=VariableTypeEnum.real,
        lowerbound=0.1,
        upperbound=10.0,
    )
    x_4 = Variable(
        name="x_4",
        symbol="x_4",
        variable_type=VariableTypeEnum.real,
        lowerbound=0.125,
        upperbound=5.0,
    )

    P = 6000
    L = 14
    E = 30 * 1e6
    G = 12 * 1e6
    tauMax = 13600
    sigmaMax = 30000

    M = f"{P} * ({L} + (x_2 / 2))"
    R = "Sqrt(((x_2 * x_2) / 4.0) + ((x_1 + x_3) / 2.0)**2)"
    J = "2 * Sqrt(2) * x_1 * x_2 * ((x_2 * x_2) / 12.0) + ((x_1 + x_3) / 2.0)**2"

    tauDashDash = f"({M} * {R}) / {J}"
    tauDash = f"{P} / (Sqrt(2) * x_1 * x_2)" 
    tau = f"Sqrt({tauDash} * {tauDash} + ((2 * {tauDash} * {tauDashDash} * x_2) / (2 * {R})) + ({tauDashDash} * {tauDashDash}))"
    sigma = f"(6 * {P} * {L}) / (x_4 * x_3 * x_3)"
    tmpVar = f"4.013 * {E} * Sqrt((x_3 * x_3 * x_4 * x_4 * x_4 * x_4 * x_4 * x_4) / 36.0) / ({L} * {L})"
    tmpVar2 = f"(x_3 / (2 * {L})) * Sqrt({E} / (4 * {G}))"
    PC = f"{tmpVar} * (1 - {tmpVar2})"

    f_1 = Objective(
        name="f_1",
        symbol="f_1",
        func="(1.10471 * x_1 * x_1 * x_2) + (0.04811 * x_3 * x_4) * (14.0 + x_2)",
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=False,
        is_convex=False,
        is_twice_differentiable=True,
    )
    f_2 = Objective(
        name="f_2",
        symbol="f_2",
        func=f"(4 * {P} * {L} * {L} * {L}) / ({E} * x_4 * x_3 * x_3 * x_3)",
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=False,
        is_convex=False,
        is_twice_differentiable=True,
    )
    f_3 = Objective(
        name="f_3",
        symbol="f_3",
        func=f"Max({tau}-{tauMax},0) + Max({sigma} - {sigmaMax},0) + Max(x_1 - x_4, 0) + Max({P} - {PC}, 0)",
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=False,
        is_convex=False,
        is_twice_differentiable=True,
    )

    return Problem(
        name="Welded beam design",
        description="The welded beam design problem",
        variables=[x_1, x_2, x_3, x_4],
        objectives=[f_1, f_2, f_3],
    )

def re33() -> Problem:
    '''
    Disc brake design.
    '''

    x_1 = Variable(
            name="x_1",
            symbol="x_1",
            variable_type=VariableTypeEnum.real,
            lowerbound=55,
            upperbound=80,
    )
    x_2 = Variable(
                name="x_2",
                symbol="x_2",
                variable_type=VariableTypeEnum.real,
                lowerbound=75,
                upperbound=110,
    )
    x_3 = Variable(
                    name="x_3",
                    symbol="x_3",
                    variable_type=VariableTypeEnum.real,
                    lowerbound=1000,
                    upperbound=3000,
    )
    x_4 = Variable(
                    name="x_4",
                    symbol="x_4",
                    variable_type=VariableTypeEnum.real,
                    lowerbound=11,
                    upperbound=20,
    )

    f_1 = Objective(
        name="f_1",
        symbol="f_1",
        func="4.9 * 1e-5 * (x_2 * x_2 - x_1 * x_1) * (x_4 - 1.0)",
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=False,
        is_convex=False,
        is_twice_differentiable=True,
    )    
    f_2 = Objective(
        name="f_2",
        symbol="f_2",
        func="((9.82 * 1e6) * (x_2 * x_2 - x_1 * x_1))" 
        "/ (x_3 * x_4 * (x_2 * x_2 * x_2 - x_1 * x_1 * x_1))",
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=False,
        is_convex=False,
        is_twice_differentiable=True,
    ) 
    f_3 = Objective(
        name="f_3",
        symbol="f_3",
        func="Max(20 - (x_2 - x_1), 0)"
        "+ Max((x_3 / (3.14 * (x_2 * x_2 - x_1 * x_1))) - 0.4, 0)"
        "+ Max((2.22 * 1e-3 * x_3 * (x_2 * x_2 * x_2 - x_1 * x_1 * x_1)) / ((x_2 * x_2 - x_1 * x_1)**2) - 1.0, 0)"
        "+ Max(900 - (2.66 * 1e-2 * x_3 * x_4 * (x_2 * x_2 * x_2 - x_1 * x_1 * x_1)) / (x_2 * x_2 - x_1 * x_1), 0)",
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=False,
        is_convex=False,
        is_twice_differentiable=True,
    ) 

    return Problem(
        name="Disc brake design.",
        description="The disc brake design problem.",
        variables=[x_1, x_2, x_3, x_4],
        objectives=[f_1, f_2, f_3],
    )


def re42() -> Problem:
    '''
    Conceptual marine design.
    '''
    # TODO: there are major bugs in this implementation, doesn't give the same results as the original for objective functions 1 and 3
    # (partially differing results for f_4)
    x_1 = Variable(
        name="x_1",
        symbol="x_1",
        variable_type=VariableTypeEnum.real,
        lowerbound=150.0,
        upperbound=274.32,
    )
    x_2 = Variable(
        name="x_2",
        symbol="x_2",
        variable_type=VariableTypeEnum.real,
        lowerbound=20.0,
        upperbound=32.31,
    )
    x_3 = Variable(
        name="x_3",
        symbol="x_3",
        variable_type=VariableTypeEnum.real,
        lowerbound=13.0,
        upperbound=25.0,
    )
    x_4 = Variable(
        name="x_4",
        symbol="x_4",
        variable_type=VariableTypeEnum.real,
        lowerbound=10.0,
        upperbound=11.71,
    )
    x_5 = Variable(
        name="x_5",
        symbol="x_5",
        variable_type=VariableTypeEnum.real,
        lowerbound=14.0,
        upperbound=18.0,
    )
    x_6 = Variable(
        name="x_6",
        symbol="x_6",
        variable_type=VariableTypeEnum.real,
        lowerbound=0.63,
        upperbound=0.75,
    )

    x_L = f"x_1"
    x_B = f"x_2"
    x_D = f"x_3"
    x_T = f"x_4"
    x_Vk = f"x_5"
    x_CB = f"x_6"

    displacement = f"1.025 * {x_L} * {x_B} * {x_T} * {x_CB}"
    #V = f"0.5144 * {x_Vk}"
    #g = "9.8065"
    Fn = f"(0.5144 * {x_Vk}) / (9.8065 * {x_L})**0.5"
    a = f"((4977.06 * {x_CB} * {x_CB}) - (8105.61 * {x_CB}) + 4456.51)"
    b = f"((-10847.2 * {x_CB} * {x_CB}) + (12817.0 * {x_CB}) - 6960.32)"

    power = f"(({displacement})**(2.0/3.0) * {x_Vk}**3.0) / ({a} + ({b} * {Fn}))"
    outfit_weight = f"1.0 * {x_L}**0.8 * {x_B}**0.6 * {x_D}**0.3 * {x_CB}**0.1"
    steel_weight = f"0.034 * {x_L}**1.7 * {x_B}**0.7 * {x_D}**0.4 * {x_CB}**0.5"
    machinery_weight = f"0.17 * ({power})**0.9"
    light_ship_weight = f"{steel_weight} + {outfit_weight} + {machinery_weight}"

    ship_cost = f"0.26 * (2000.0 * ({steel_weight})**0.85  + 3500.0 * {outfit_weight} + 2400.0 * {power}**0.8)"
    capital_costs = f"(0.2 * {ship_cost})"

    DWT = f"({displacement} - {light_ship_weight})"

    running_costs = f"40000.0 * {DWT}**0.3"

    #round_trip_miles = "5000.0"
    sea_days = f"(5000.0 / 24.0 * {x_Vk})"
    #handling_rate = "8000.0"

    daily_consumption = f"(0.00456 * {power} + 0.2)"
    #fuel_price = "100.0"
    fuel_cost = f"1.05 * {daily_consumption} * {sea_days} * 100.0"
    port_cost = f"6.3 * {DWT}**0.8"

    fuel_carried = f"({daily_consumption} * ({sea_days} + 5.0))"
    miscellaneous_DWT = f"2.0 * {DWT}**0.5"

    cargo_DWT = f"({DWT} - {fuel_carried} - {miscellaneous_DWT})"
    port_days = f"2.0 * (({cargo_DWT} / 8000.0) + 0.5)"
    RTPA = f"350.0 / ({sea_days} + {port_days})"

    voyage_costs = f"({fuel_cost} + {port_cost}) * {RTPA}"
    annual_costs = f"({capital_costs} + {running_costs} + {voyage_costs})"
    print("annual_costs", annual_costs)
    annual_cargo = f"({cargo_DWT} * {RTPA})"
    print("annual_cargo", annual_cargo)

    KB = f"0.53 * {x_T}"
    BMT = f"((0.085 * {x_CB} - 0.002) * {x_B} * {x_B}) / ({x_T} * {x_CB})"
    KG = f"(1.0 + 0.52 * {x_D})"


    f_1 = Objective(
        name="f_1",
        symbol="f_1",
        func=f"{annual_costs}/{annual_cargo}",
        objective_type=ObjectiveTypeEnum.analytical,
    )   
    f_2 = Objective(
        name="f_2",
        symbol="f_2",
        func=f"{light_ship_weight}",
        objective_type=ObjectiveTypeEnum.analytical,
    )          
    f_3 = Objective(
        name="f_3",
        symbol="f_3",
        func=f"-{annual_cargo}",
        objective_type=ObjectiveTypeEnum.analytical,
    )    
    f_4 = Objective(
        name="f_4",
        symbol="f_4",
        func=f"Max(6.0 - ({x_L} / {x_B}), 0)"
        f"+ Max({x_L}/{x_D} - 15.0, 0)"
        f"+ Max({x_L}/{x_T} - 19.0, 0)"
        f"+ Max({x_T} - 0.45*{DWT}**0.31, 0)"
        f"+ Max({x_T} - 0.7 * {x_D} - 0.7, 0)"
        f"+ Max({DWT} - 500000.0, 0)"
        f"+ Max(3000 - {DWT}, 0)"
        f"+ Max({Fn} - 0.32, 0)"
        f"+ Max((0.07 * {x_B}) - ({KB} + {BMT} - {KG}), 0)",
        objective_type=ObjectiveTypeEnum.analytical,
    )    


    return Problem(
        name="Conceptual marine design",
        description="The Conceptual marine design problem.",
        variables=[x_1, x_2, x_3, x_4, x_5, x_6],
        objectives=[f_1, f_2, f_3, f_4],
    )
