/* eslint-disable import/no-anonymous-default-export */
import React, { useRef, useLayoutEffect } from 'react';
import * as am5 from "@amcharts/amcharts5";
import * as am5xy from "@amcharts/amcharts5/xy";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";


export default (props) => {
    const chartRef = useRef(null);

    useLayoutEffect(() => {
        let data = props.chartData
        let displayVariables = props.chartVariables

        // Create root element
        // https://www.amcharts.com/docs/v5/getting-started/#Root_element
        let root = am5.Root.new("chartdiv");

        // Set themes
        // https://www.amcharts.com/docs/v5/concepts/themes/
        root.setThemes([
            am5themes_Animated.new(root)
        ]);

        // Create chart
        // https://www.amcharts.com/docs/v5/charts/xy-chart/
        let chart = root.container.children.push(
            am5xy.XYChart.new(root, {
                layout: root.verticalLayout
            })
        );

        // Create axes
        // https://www.amcharts.com/docs/v5/charts/xy-chart/axes/

        // Y axes
        let yAxis = chart.yAxes.push(
            am5xy.ValueAxis.new(root, {
                renderer: am5xy.AxisRendererY.new(root, {})
            })
        );

        // X axis
        let xAxis = chart.xAxes.push(
            am5xy.CategoryDateAxis.new(root, {
                renderer: am5xy.AxisRendererX.new(root, {}),
                categoryField: 'date'
            })
        );
        xAxis.data.setAll(data)

        // Add series
        // https://www.amcharts.com/docs/v5/charts/xy-chart/series/
        for (const col of displayVariables) {
            let series = chart.series.push(am5xy.SmoothedYLineSeries.new(root, {
                name: col,
                xAxis: xAxis,
                yAxis: yAxis,
                valueYField: col,
                categoryXField: "date",
                legendValueText: "{valueY}"
            }))

            
            series.data.processor = am5.DataProcessor.new(root, {
                dateFields: ["date"]
            });

            series.data.setAll(data)
        }

        // Add series tooltips
        // https://www.amcharts.com/docs/v5/charts/xy-chart/series/#Tooltips



        // Add legend to axis header
        // https://www.amcharts.com/docs/v5/charts/xy-chart/axes/axis-headers/
        // https://www.amcharts.com/docs/v5/charts/xy-chart/legend-xy-series/
        let legend = chart.children.push(am5.Legend.new(root, {
            x: am5.percent(50),
            centerX: am5.percent(50)
        }));
        legend.data.setAll(chart.series.values);


        // Add cursor
        // https://www.amcharts.com/docs/v5/charts/xy-chart/cursor/
        chart.set("cursor", am5xy.XYCursor.new(root, {}));


        // Add scrollbar
        // https://www.amcharts.com/docs/v5/charts/xy-chart/scrollbars/



        // For Painting Chart Updates
        chartRef.current = chart;
        return () => {
            root.dispose();
        };
    }, [props.chartData, props.chartVariables]);

    useLayoutEffect(() => {
        chartRef.current.set("paddingRight", props.paddingRight);
    }, [props.paddingRight]);

    return (
        <div id="chartdiv" style={{ width: "100%", height: "480px" }}></div>
    );
}