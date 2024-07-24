import logging

from ra2ce.analysis.losses.risk_calculation.risk_calculation_base import (
    RiskCalculationBase,
)


class RiskCalculationTriangleToNullYear(RiskCalculationBase):
    """
    In this mode, an extra data point with zero damage is added at some distance from the smallest known RP,
    and the area of the Triangle this creates is also calculated
    """

    def _rework_damage_data(self):
        """
        Rework the damage data to make it suitable for integration (risk calculation) in triangle_to_null_year mode
        """
        if (
            self.risk_calculation_year >= self._min_return_period
            and self.risk_calculation_year != 0
        ):
            raise ValueError(
                """
            RA2CE cannot calculate risk in 'triangle_to_null' mode if 
            Return period of the triangle ({}) >= smallest available return period ({})
            Use 'default' mode or 'cut_from' instead.
                                """.format(
                    self.risk_calculation_year, self._min_return_period
                )
            )

        if self.risk_calculation_year == 0:
            logging.warning(
                "Available lane data cannot simply be converted to float/int, RA2CE will try a clean-up."
            )
            self.risk_calculation_year = 1

        self._to_integrate[float("inf")] = self._to_integrate[self._max_return_period]

        # At the return period of the self.risk_calculation_year, set all damage values to zero
        self._to_integrate[self.risk_calculation_year] = 0

        self._to_integrate = self._to_integrate.sort_index(
            axis="columns", ascending=False
        )  # from large to small RP

        self._to_integrate = self._to_integrate.fillna(0)

        logging.info(
            """Risk calculation runs in 'triangle to null' mode. 
                                Assumptions:
                                    - for all return periods > max RP{}, damage = dam_RP{}
                                    - at the end of the triangle {}, damage = 0

                                """.format(
                self._max_return_period,
                self._max_return_period,
                self.risk_calculation_year,
            )
        )
