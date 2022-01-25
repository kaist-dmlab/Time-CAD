/* eslint-disable import/no-anonymous-default-export */
import React, { useRef, useLayoutEffect } from 'react';
import * as am5 from "@amcharts/amcharts5";
import * as am5xy from "@amcharts/amcharts5/xy";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";


export default (props) => {
    const chartRef = useRef(null);

    useLayoutEffect(() => {

        let root = am5.Root.new("chartdiv");

        root.setThemes([
            am5themes_Animated.new(root)
        ]);

        let chart = root.container.children.push(
            am5xy.XYChart.new(root, {
                panY: false,
                layout: root.verticalLayout
            })
        );

        // Define data
        let data = [{
            category: "Research",
            value1: 1000,
            value2: 588
        }, {
            category: "Marketing",
            value1: 1200,
            value2: 1800
        }, {
            category: "Sales",
            value1: 850,
            value2: 1230
        }];

        // Create Y-axis
        let yAxis = chart.yAxes.push(
            am5xy.ValueAxis.new(root, {
                renderer: am5xy.AxisRendererY.new(root, {})
            })
        );

        // Create X-Axis
        let xAxis = chart.xAxes.push(
            am5xy.CategoryAxis.new(root, {
                renderer: am5xy.AxisRendererX.new(root, {}),
                categoryField: "date"
            })
        );
        xAxis.data.setAll(props.chartData);

        // Create series
        let series1 = chart.series.push(
            am5xy.SmoothedXLineSeries.new(root, {
                name: "Series",
                xAxis: xAxis,
                yAxis: yAxis,
                valueYField: "value",
                categoryXField: "date",
            })
        );
        series1.data.setAll(props.chartData);

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

        let series = chart.series.push(am5xy.LineSeries.new(root, {
            name: "Series",
            xAxis: xAxis,
            yAxis: yAxis,
            valueYField: "value",
            valueXField: "date",
            tooltip: am5.Tooltip.new(root, {
                labelText: "{valueY}"
            })
        }));

        // Add legend
        let legend = chart.children.push(am5.Legend.new(root, {}));
        legend.data.setAll(chart.series.values);

        // Add cursor
        chart.set("cursor", am5xy.XYCursor.new(root, {}));
        chart.set("scrollbarX", am5.Scrollbar.new(root, {
            orientation: "horizontal"
        }));

        // series.data.setAll(props.chartData);

        chartRef.current = chart;

        return () => {
            root.dispose();
        };
    }, []);

    useLayoutEffect(() => {
        chartRef.current.set("paddingRight", props.paddingRight);
    }, [props.paddingRight]);

    return (
        <div id="chartdiv" style={{ width: "100%", height: "400px" }}></div>
    );
}