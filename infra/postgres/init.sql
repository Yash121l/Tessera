-- Tessera database schema
-- Orders, fills, and positions for the trading system

CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(5) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(16) NOT NULL,
    quantity NUMERIC(18, 8) NOT NULL,
    price NUMERIC(18, 8),
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    exchange VARCHAR(16) NOT NULL,
    exchange_order_id VARCHAR(128),
    strategy_id VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES orders(id),
    price NUMERIC(18, 8) NOT NULL,
    quantity NUMERIC(18, 8) NOT NULL,
    fee NUMERIC(18, 8) NOT NULL DEFAULT 0,
    fee_currency VARCHAR(16),
    filled_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(32) NOT NULL UNIQUE,
    side VARCHAR(5) NOT NULL CHECK (side IN ('long', 'short', 'flat')),
    size NUMERIC(18, 8) NOT NULL DEFAULT 0,
    entry_price NUMERIC(18, 8),
    unrealized_pnl NUMERIC(18, 8) DEFAULT 0,
    realized_pnl NUMERIC(18, 8) DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
