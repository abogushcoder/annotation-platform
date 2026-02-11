import json
import re
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
        {'name': 'currentItemName', 'label': 'Current Item', 'type': 'text'},
        {'name': 'newItemName', 'label': 'New Item', 'type': 'text'},
        {'name': 'newQuantity', 'label': 'New Quantity', 'type': 'number'},
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
    'end_call': [],
    'send_menu_link': [
        {'name': 'customerPhone', 'label': 'Customer Phone', 'type': 'text'},
    ],
}

# Tools that have purpose-built card templates
KNOWN_TOOLS = {
    'create_order', 'cancel_order', 'remove_item', 'modify_item',
    'check_availability', 'create_reservation', 'end_call',
    'get_past_orders', 'send_menu_link',
}

# Tools that have purpose-built form templates
FORM_TOOLS = {
    'create_order', 'cancel_order', 'remove_item', 'modify_item',
    'check_availability', 'create_reservation',
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


@register.simple_tag
def dict_items(d):
    """Safely iterate over a dict's items in templates.

    Django template resolution of `mydict.items` will return the value of
    a key named "items" if it exists, instead of calling .items().  This
    tag bypasses that by calling dict.items() directly.
    """
    if not isinstance(d, dict):
        return []
    result = []
    for key, value in d.items():
        display_value = json.dumps(value, indent=2) if isinstance(value, (list, dict)) else value
        result.append({'key': key, 'value': value, 'display_value': display_value})
    return result


def _format_phone(phone):
    """Format a phone number for display, e.g. '6464014800' -> '(646) 401-4800'."""
    if not phone:
        return ''
    digits = re.sub(r'\D', '', str(phone))
    if digits.startswith('1') and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10:
        return f'({digits[:3]}) {digits[3:6]}-{digits[6:]}'
    return str(phone)


def _format_money(value):
    """Format a numeric value as money string like '$12.99'."""
    if value is None:
        return ''
    try:
        return f'${float(value):.2f}'
    except (ValueError, TypeError):
        return str(value)


def _parse_items(items_data):
    """Parse items from tool call args into a template-friendly list."""
    if not isinstance(items_data, list):
        return []
    result = []
    for item in items_data:
        if not isinstance(item, dict):
            continue
        modifiers = []
        raw_mods = item.get('modifiers', [])
        if isinstance(raw_mods, list):
            for m in raw_mods:
                if isinstance(m, dict):
                    modifiers.append(m.get('name', str(m)))
                elif isinstance(m, str):
                    modifiers.append(m)
        result.append({
            'name': item.get('itemName', item.get('name', '???')),
            'quantity': item.get('quantity', 1),
            'modifiers': modifiers,
            'special': item.get('specialInstructions', ''),
        })
    return result


def _format_date(date_str):
    """Format a date string like '2026-02-11' to 'Feb 11, 2026'."""
    if not date_str:
        return ''
    try:
        from datetime import datetime
        dt = datetime.strptime(str(date_str), '%Y-%m-%d')
        return dt.strftime('%b %d, %Y')
    except (ValueError, TypeError):
        return str(date_str)


def _format_time(time_str):
    """Format a time string like '19:30' to '7:30 PM'."""
    if not time_str:
        return ''
    try:
        from datetime import datetime
        # Handle various formats
        for fmt in ('%H:%M', '%H:%M:%S'):
            try:
                dt = datetime.strptime(str(time_str), fmt)
                return dt.strftime('%-I:%M %p')
            except ValueError:
                continue
        return str(time_str)
    except (ValueError, TypeError):
        return str(time_str)


@register.simple_tag
def get_tool_card_template(tc):
    """Return the template path for a tool call's display card."""
    tool = tc.tool_name
    if tool in KNOWN_TOOLS:
        return f'conversations/partials/tools/_card_{tool}.html'
    return 'conversations/partials/tools/_card_generic.html'


@register.simple_tag
def get_tool_form_template(tc):
    """Return the template path for a tool call's edit form."""
    tool = tc.tool_name
    if tool in FORM_TOOLS:
        return f'conversations/partials/tools/_form_{tool}.html'
    return 'conversations/partials/tools/_form_generic.html'


@register.simple_tag
def get_tool_display_data(tc):
    """Pre-process tool call args and response into template-friendly data."""
    args = tc.display_args
    resp = tc.response_body if isinstance(tc.response_body, dict) else {}
    tool = tc.tool_name

    # Determine error state: status_code may be None even for successful calls
    has_error = bool(tc.error_message)
    error_message = tc.error_message or ''
    # Also detect errors from raw response bodies
    if not has_error and isinstance(resp.get('raw'), str) and 'Error' in resp.get('raw', ''):
        has_error = True
        error_message = resp['raw']

    data = {
        'tool_name': tool,
        'raw_args': args,
        'raw_response': resp,
        'has_error': has_error,
        'error_message': error_message,
    }

    if tool == 'create_order':
        data['customer_name'] = args.get('customerName', '')
        data['customer_phone'] = _format_phone(args.get('customerPhone', ''))
        data['items'] = _parse_items(args.get('items', []))
        data['special_instructions'] = args.get('specialInstructions', '')
        # Response data may be nested under resp['order']
        order = resp.get('order', {}) if isinstance(resp.get('order'), dict) else {}
        data['result'] = {
            'success': resp.get('success', False),
            'order_number': order.get('orderNumber', resp.get('orderNumber', resp.get('orderId', ''))),
            'total': _format_money(order.get('total', resp.get('total', resp.get('orderTotal')))),
            'wait_minutes': order.get('estimatedWaitMinutes', resp.get('estimatedWaitMinutes', resp.get('waitMinutes', ''))),
            'message': resp.get('message', ''),
        }

    elif tool == 'cancel_order':
        data['order_id'] = args.get('orderId', '')
        data['reason'] = args.get('reason', '')
        data['result'] = {
            'success': resp.get('success', False),
            'message': resp.get('message', ''),
        }

    elif tool == 'modify_item':
        data['order_id'] = args.get('orderId', '')
        # Support both old (itemName/modifications) and new (currentItemName/newItemName) field names
        data['current_item_name'] = args.get('currentItemName', args.get('itemName', ''))
        data['new_item_name'] = args.get('newItemName', '')
        data['new_quantity'] = args.get('newQuantity', '')
        data['modifications'] = args.get('modifications', '')
        data['result'] = {
            'success': resp.get('success', False),
            'message': resp.get('message', ''),
        }

    elif tool == 'remove_item':
        data['order_id'] = args.get('orderId', '')
        data['item_name'] = args.get('itemName', '')
        data['result'] = {
            'success': resp.get('success', False),
            'message': resp.get('message', ''),
        }

    elif tool == 'check_availability':
        data['date'] = _format_date(args.get('date', ''))
        data['time'] = _format_time(args.get('time', ''))
        data['party_size'] = args.get('partySize', '')
        available = resp.get('available', resp.get('isAvailable'))
        data['result'] = {
            'available': available,
            'message': resp.get('message', ''),
            'time_slots': resp.get('availableTimeSlots', resp.get('timeSlots', [])),
        }

    elif tool == 'create_reservation':
        data['customer_name'] = args.get('customerName', '')
        data['customer_phone'] = _format_phone(args.get('customerPhone', ''))
        data['party_size'] = args.get('partySize', '')
        data['date'] = _format_date(args.get('date', ''))
        data['time'] = _format_time(args.get('time', ''))
        data['special_requests'] = args.get('specialRequests', '')
        # Response data may be nested under resp['reservation']
        reservation = resp.get('reservation', {}) if isinstance(resp.get('reservation'), dict) else {}
        data['result'] = {
            'success': resp.get('success', False),
            'confirmation_code': reservation.get('confirmationCode', resp.get('confirmationCode', resp.get('confirmation_code', ''))),
            'table': reservation.get('tableName', resp.get('tableName', resp.get('table', ''))),
            'message': resp.get('message', ''),
        }

    elif tool == 'end_call':
        data['reason'] = args.get('reason', args.get('message', ''))

    elif tool == 'get_past_orders':
        data['customer_phone'] = _format_phone(args.get('customerPhone', ''))
        orders = resp.get('orders', resp.get('pastOrders', []))
        data['orders'] = orders if isinstance(orders, list) else []
        data['result'] = {
            'message': resp.get('message', ''),
        }

    elif tool == 'send_menu_link':
        data['customer_phone'] = _format_phone(args.get('customerPhone', ''))
        data['result'] = {
            'success': resp.get('success', False),
            'message': resp.get('message', ''),
        }

    else:
        # Generic: build key-value pairs
        pairs = []
        for key, value in args.items():
            display_value = json.dumps(value, indent=2) if isinstance(value, (list, dict)) else str(value)
            pairs.append({'key': key, 'value': display_value})
        data['pairs'] = pairs
        resp_pairs = []
        for key, value in resp.items():
            display_value = json.dumps(value, indent=2) if isinstance(value, (list, dict)) else str(value)
            resp_pairs.append({'key': key, 'value': display_value})
        data['resp_pairs'] = resp_pairs

    return data
