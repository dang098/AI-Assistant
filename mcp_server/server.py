import datetime
import uuid

from fastmcp import FastMCP
from supabase import Client, create_client

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)
mcp = FastMCP("EcommerceOperationsServer")


def get_supabase_client() -> Client:
    if not settings.supabase_url or not settings.supabase_key:
        raise ValueError("Chưa cấu hình SUPABASE_URL hoặc SUPABASE_KEY trong .env")
    return create_client(settings.supabase_url, settings.supabase_key)


@mcp.tool()
def check_inventory(query: str) -> dict:
    """Tra cứu thông tin sản phẩm, giá bán và số lượng tồn kho theo tên hoặc mã SKU."""
    client = get_supabase_client()
    logger.info(f"Tool check_inventory: query='{query}'")

    search_term = query.replace(",", "").replace("%", "").strip()
    res = (
        client.table("products")
        .select("id, sku, name, price, stock_quantity, category")
        .or_(f"name.ilike.%{search_term}%,sku.ilike.%{search_term}%")
        .limit(5)
        .execute()
    )

    if not res.data and " " in search_term:
        words = [w for w in search_term.split() if len(w) > 2]
        if words:
            first_word = words[0]
            res = (
                client.table("products")
                .select("id, sku, name, price, stock_quantity, category")
                .or_(f"name.ilike.%{first_word}%,sku.ilike.%{first_word}%")
                .limit(5)
                .execute()
            )

    if not res.data:
        return {"status": "not_found", "message": f"Không tìm thấy sản phẩm '{query}'."}

    return {"status": "success", "total_matches": len(res.data), "products": res.data}


@mcp.tool()
def create_order(
    product_id: int,
    quantity: int,
    customer_name: str,
    phone: str,
    full_address: str,
) -> dict:
    """Tạo đơn hàng mới và tự động cập nhật trừ tồn kho trong cơ sở dữ liệu."""
    client = get_supabase_client()
    logger.info(f"Tool create_order: product_id={product_id}, qty={quantity}")

    if quantity <= 0:
        return {"status": "error", "message": "Số lượng đặt hàng phải lớn hơn 0."}

    p_res = (
        client.table("products")
        .select("id, name, price, stock_quantity")
        .eq("id", product_id)
        .execute()
    )
    if not p_res.data:
        return {
            "status": "error",
            "message": f"Sản phẩm ID {product_id} không tồn tại.",
        }

    product = p_res.data[0]
    current_stock = product.get("stock_quantity") or 0
    if current_stock < quantity:
        return {
            "status": "error",
            "message": f"'{product['name']}' chỉ còn {current_stock} sản phẩm.",
        }

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    order_sn = f"ORD_{now_utc.strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:4].upper()}"
    total_amount = float(product["price"]) * quantity

    client.table("orders").insert(
        {
            "order_sn": order_sn,
            "order_status": "pending",
            "customer_name": customer_name,
            "phone": phone,
            "full_address": full_address,
            "total_amount": total_amount,
            "is_stock_deducted": True,
            "ordered_at": now_utc.isoformat(),
        }
    ).execute()

    new_stock = current_stock - quantity
    try:
        update_res = (
            client.table("products")
            .update({"stock_quantity": new_stock})
            .eq("id", product_id)
            .gte("stock_quantity", quantity)
            .execute()
        )
        if not update_res.data:
            client.table("orders").delete().eq("order_sn", order_sn).execute()
            return {
                "status": "error",
                "message": "Sản phẩm vừa hết hàng do có khách khác đặt trước.",
            }
    except (ValueError, KeyError, RuntimeError) as e:
        client.table("orders").delete().eq("order_sn", order_sn).execute()
        logger.error(f"Lỗi trừ kho, đã hủy đơn {order_sn}: {e}")
        return {
            "status": "error",
            "message": "Lỗi hệ thống khi cập nhật kho, giao dịch đã được hủy an toàn.",
        }

    return {
        "status": "success",
        "order_sn": order_sn,
        "product_name": product["name"],
        "quantity": quantity,
        "total_amount": total_amount,
        "remaining_stock": new_stock,
    }


@mcp.tool()
def update_order_info(
    order_sn: str,
    new_address: str | None = None,
    new_phone: str | None = None,
) -> dict:
    """Cập nhật địa chỉ nhận hàng hoặc số điện thoại của đơn hàng đã tạo."""
    client = get_supabase_client()
    logger.info(f"Tool update_order_info: order_sn='{order_sn}'")

    order_res = client.table("orders").select("id").eq("order_sn", order_sn).execute()
    if not order_res.data:
        return {"status": "error", "message": f"Không tìm thấy đơn hàng '{order_sn}'."}

    update_payload = {}
    if new_address:
        update_payload["full_address"] = new_address
    if new_phone:
        update_payload["phone"] = new_phone

    if not update_payload:
        return {
            "status": "warning",
            "message": "Không có thông tin nào được yêu cầu thay đổi.",
        }

    client.table("orders").update(update_payload).eq("order_sn", order_sn).execute()
    return {"status": "success", "order_sn": order_sn, "updated": update_payload}


if __name__ == "__main__":
    mcp.run()
