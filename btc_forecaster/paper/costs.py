"""Versioned, venue-neutral transaction-cost assumptions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    version: str
    maker_fee: float
    taker_fee: float
    spread: float
    slippage: float
    funding: float = 0.0
    borrow_cost: float = 0.0

    def round_trip_fraction(self, *, maker_entry: bool = False) -> float:
        values = (
            self.maker_fee,
            self.taker_fee,
            self.spread,
            self.slippage,
            self.funding,
            self.borrow_cost,
        )
        if any(value < 0 for value in values):
            raise ValueError("cost assumptions cannot be negative")
        entry_fee = self.maker_fee if maker_entry else self.taker_fee
        return (
            entry_fee
            + self.taker_fee
            + self.spread
            + 2 * self.slippage
            + self.funding
            + self.borrow_cost
        )
