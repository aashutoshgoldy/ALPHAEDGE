import math

class OptionsPricingModel:
    """
    Pure Python Black-Scholes-Merton model for Options Pricing, Greeks, and Probability of Profit (POP).
    Zero external dependencies (uses standard library math).
    Accurately models Indian Index & Stock options (European style).
    """

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Standard Normal Cumulative Distribution Function using math.erf"""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        """Standard Normal Probability Density Function"""
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

    @classmethod
    def d1(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """S=Spot, K=Strike, T=Time to expiry in years, r=Risk-free rate, sigma=Volatility"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    @classmethod
    def d2(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        return cls.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    @classmethod
    def calculate_price(cls, S: float, K: float, T: float, r: float = 0.07, sigma: float = 0.15, option_type: str = 'CE') -> float:
        """
        Calculate theoretical option premium.
        r = 7% (Indian risk-free rate approx)
        sigma = 15% (Average India VIX / implied volatility)
        T = Days / 365
        """
        if T <= 0.0001:
            if option_type == 'CE':
                return max(0.0, S - K)
            else:
                return max(0.0, K - S)

        d1_val = cls.d1(S, K, T, r, sigma)
        d2_val = cls.d2(S, K, T, r, sigma)

        if option_type == 'CE':
            price = S * cls.norm_cdf(d1_val) - K * math.exp(-r * T) * cls.norm_cdf(d2_val)
        else:
            price = K * math.exp(-r * T) * cls.norm_cdf(-d2_val) - S * cls.norm_cdf(-d1_val)

        return max(0.05, float(price))

    @classmethod
    def calculate_greeks(cls, S: float, K: float, T: float, r: float = 0.07, sigma: float = 0.15, option_type: str = 'CE') -> dict:
        """Calculate Delta, Theta, Gamma, Vega"""
        if T <= 0.0001 or sigma <= 0:
            delta = 1.0 if (option_type == 'CE' and S > K) else (-1.0 if (option_type == 'PE' and S < K) else 0.0)
            return {'delta': delta, 'theta': 0.0, 'gamma': 0.0, 'vega': 0.0}

        d1_val = cls.d1(S, K, T, r, sigma)
        d2_val = cls.d2(S, K, T, r, sigma)
        pdf_d1 = cls.norm_pdf(d1_val)

        if option_type == 'CE':
            delta = cls.norm_cdf(d1_val)
            theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * cls.norm_cdf(d2_val)) / 365.0
        else:
            delta = cls.norm_cdf(d1_val) - 1.0
            theta = (- (S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * cls.norm_cdf(-d2_val)) / 365.0

        gamma = pdf_d1 / (S * sigma * math.sqrt(T))
        vega = (S * math.sqrt(T) * pdf_d1) / 100.0  # per 1% change in IV

        return {
            'delta': round(float(delta), 4),
            'theta': round(float(theta), 2),
            'gamma': round(float(gamma), 6),
            'vega': round(float(vega), 2)
        }

    @classmethod
    def calculate_pop(cls, S: float, breakeven: float, T: float, sigma: float = 0.15, trade_type: str = 'BUY_CALL') -> float:
        """
        Probability of Profit (POP).
        Calculates theoretical probability that the spot price finishes beyond breakeven at expiry.
        """
        if T <= 0.0001 or sigma <= 0 or S <= 0 or breakeven <= 0:
            return 50.0

        mu = 0.07  # expected drift rate
        drift = (mu - 0.5 * sigma ** 2) * T
        std = sigma * math.sqrt(T)

        z = (math.log(breakeven / S) - drift) / std

        if trade_type in ['BUY_CALL', 'LONG_STOCK']:
            pop = (1.0 - cls.norm_cdf(z)) * 100.0
        elif trade_type in ['BUY_PUT', 'SHORT_STOCK', 'SELL_CALL', 'CREDIT_SPREAD_CALL']:
            pop = cls.norm_cdf(z) * 100.0
        elif trade_type in ['SELL_PUT', 'CREDIT_SPREAD_PUT']:
            pop = (1.0 - cls.norm_cdf(z)) * 100.0
        else:
            pop = 50.0

        return max(1.0, min(99.0, round(float(pop), 2)))
