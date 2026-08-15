CLASS_REGISTRY = {}


def _register():
    from .gorilla_file_system import GorillaFileSystem
    from .math_api import MathAPI
    from .message_api import MessageAPI
    from .posting_api import TwitterAPI
    from .ticket_api import TicketAPI
    from .trading_bot import TradingBot
    from .travel_booking import TravelAPI
    from .vehicle_control import VehicleControlAPI

    CLASS_REGISTRY.update({
        "GorillaFileSystem": GorillaFileSystem,
        "MathAPI": MathAPI,
        "MessageAPI": MessageAPI,
        "TwitterAPI": TwitterAPI,
        "TicketAPI": TicketAPI,
        "TradingBot": TradingBot,
        "TravelAPI": TravelAPI,
        "VehicleControlAPI": VehicleControlAPI,
    })


_register()
