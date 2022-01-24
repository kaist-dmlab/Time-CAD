/* eslint-disable import/no-anonymous-default-export */
import React, { useRef, useLayoutEffect } from 'react';
import * as am5 from "@amcharts/amcharts5";
import * as am5xy from "@amcharts/amcharts5/xy";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";


export default ({ chartData, setCheckedList }) => {
    const [data, setData] = React.useState([])

    React.useEffect(() => {
        setData(chartData)
    }, [chartData]);

    useLayoutEffect(() => {

        let root = am5.Root.new("chartdiv");

        root.setThemes([
            am5themes_Animated.new(root)
        ]);

        let chart = root.container.children.push(
            am5xy.XYChart.new(root, {
                panX: true,
                panY: true,
                wheelX: "panX",
                wheelY: "zoomX",
                maxTooltipDistance: 0
                // layout: root.verticalLayout
            })
        );

        // // Define data
        // let data = [{
        //     category: "Research",
        //     value1: 1000,
        //     value2: 588
        // }, {
        //     category: "Marketing",
        //     value1: 1200,
        //     value2: 1800
        // }, {
        //     category: "Sales",
        //     value1: 850,
        //     value2: 1230
        // }];

        // Create X-Axis
        let xAxis = chart.xAxes.push(
            am5xy.DateAxis.new(root, {
                maxDeviation: 0.2,
                baseInterval: {
                    timeUnit: "year",
                    count: 1
                },
                renderer: am5xy.AxisRendererX.new(root, {}),
                // categoryField: "date",
                tooltip: am5.Tooltip.new(root, {})
            })
        );
        xAxis.data.setAll(chartData);

        // Create Y-axis
        let yAxis = chart.yAxes.push(
            am5xy.ValueAxis.new(root, {
                renderer: am5xy.AxisRendererY.new(root, {})
            })
        );

        // Create series
        for (var i = 0; i < 5; i++) {
            let series = chart.series.push(am5xy.LineSeries.new(root, {
                name: "Series " + i,
                xAxis: xAxis,
                yAxis: yAxis,
                valueYField: "value",
                valueXField: "date",
                legendValueText: "{valueY}",
                tooltip: am5.Tooltip.new(root, {
                    pointerOrientation: "horizontal",
                    labelText: "{valueY}"
                })
            }));

            // date = new Date();
            // date.setHours(0, 0, 0, 0);
            // value = 0;

            // let data = generateDatas(100);
            series.data.setAll(data);

            // Make stuff animate on load
            // https://www.amcharts.com/docs/v5/concepts/animations/
            series.appear();
        }
        // let series1 = chart.series.push(
        //     am5xy.SmoothedXLineSeries.new(root, {
        //         name: "Series",
        //         xAxis: xAxis,
        //         yAxis: yAxis,
        //         valueYField: "value",
        //         categoryXField: "date"
        //     })
        // );
        // series1.data.setAll(data);

        // let series2 = chart.series.push(
        //     am5xy.ColumnSeries.new(root, {
        //         name: "Series",
        //         xAxis: xAxis,
        //         yAxis: yAxis,
        //         valueYField: "value2",
        //         categoryXField: "category"
        //     })
        // );
        // series2.data.setAll(data);

        // Add legend
        // let legend = chart.children.push(am5.Legend.new(root, {}));
        // legend.data.setAll(chart.series.values);
        let legend = chart.bottomAxesContainer.children.push(am5.Legend.new(root, {
            width: 200,
            paddingLeft: 15,
            height: am5.percent(100)
        }));
        legend.itemContainers.template.events.on("pointerover", function (e) {
            let itemContainer = e.target;

            // As series list is data of a legend, dataContext is series
            let series = itemContainer.dataItem.dataContext;

            chart.series.each(function (chartSeries) {
                if (chartSeries != series) {
                    chartSeries.strokes.template.setAll({
                        strokeOpacity: 0.15,
                        stroke: am5.color(0x000000)
                    });
                } else {
                    chartSeries.strokes.template.setAll({
                        strokeWidth: 3
                    });
                }
            })
        })
        legend.itemContainers.template.set("width", am5.p100);
        legend.valueLabels.template.setAll({
            width: am5.p100,
            textAlign: "center"
        });
        legend.data.setAll(chart.series.values);

        // Add cursor
        chart.set("cursor", am5xy.XYCursor.new(root, {}));

        return () => {
            root.dispose();
        };
    }, []);

    return (
        <div id="chartdiv" style={{ width: "100%", height: "400px" }}></div>
    );
}