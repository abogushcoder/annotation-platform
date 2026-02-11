import json
from django import template

register = template.Library()

TOOL_FIELD_DEFINITIONS = {
    'create_order': [
        {'name': 'customerName', 'label': 'Customer Name', 'type': 'text'},
        {'name': 'customerPhone', 'label': 'Customer Phone', 'type': 'text'},
        {'name': 'items', 'label': 'Items (JSON)', 'type': 'json'},
        {'name': 'specialInstructions', 'label': 'Special Instructions', 'type': 'textarea'},
    ],
    'cancel_order': [
        {'name': 'orderId', 'label': 'Order ID', 'type': 'text'},
        {'name': 'reason', 'label': 'Reason', 'type': 'textarea'},
    ],
    'remove_item': [
        {'name': 'orderId', 'label': 'Order ID', 'type': 'text'},
        {'name': 'itemName', 'label': 'Item Name', 'type': 'text'},
    ],
    'modify_item': [
        {'name': 'orderId', 'label': 'Order ID', 'type': 'text'},
        {'name': 'itemName', 'label': 'Item Name', 'type': 'text'},
        {'name': 'modifications', 'label': 'Modifications', 'type': 'textarea'},
    ],
    'check_availability': [
        {'name': 'date', 'label': 'Date', 'type': 'text'},
        {'name': 'time', 'label': 'Time', 'type': 'text'},
        {'name': 'partySize', 'label': 'Party Size', 'type': 'number'},
    ],
    'create_reservation': [
        {'name': 'customerName', 'label': 'Customer Name', 'type': 'text'},
        {'name': 'customerPhone', 'label': 'Customer Phone', 'type': 'text'},
        {'name': 'partySize', 'label': 'Party Size', 'type': 'number'},
        {'name': 'date', 'label': 'Date', 'type': 'text'},
        {'name': 'time', 'label': 'Time', 'type': 'text'},
        {'name': 'specialRequests', 'label': 'Special Requests', 'type': 'textarea'},
    ],
    'get_specials': [],
    'get_past_orders': [
        {'name': 'customerPhone', 'label': 'Customer Phone', 'type': 'text'},
    ],
}


@register.simple_tag
def get_tool_fields(tc):
    args = tc.display_args
    tool_defs = TOOL_FIELD_DEFINITIONS.get(tc.tool_name, [])

    fields = []
    for field_def in tool_defs:
        value = args.get(field_def['name'], '')
        if field_def['type'] == 'json' and isinstance(value, (list, dict)):
            value = json.dumps(value, indent=2)
        fields.append({
            'name': field_def['name'],
            'label': field_def['label'],
            'type': field_def['type'],
            'value': value,
        })

    # Also show any extra fields not in definitions
    defined_names = {f['name'] for f in tool_defs}
    for key, value in args.items():
        if key not in defined_names:
            display_value = json.dumps(value, indent=2) if isinstance(value, (list, dict)) else value
            fields.append({
                'name': key,
                'label': key,
                'type': 'json' if isinstance(value, (list, dict)) else 'text',
                'value': display_value,
            })

    return fields
