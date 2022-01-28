/* eslint-disable import/no-anonymous-default-export */
import React from 'react'
import { Line, Column } from '@ant-design/plots'
import moment from 'moment';

export default ({ chartData, threshold }) => {
    const [data, setData] = React.useState([])
    const [thresholdScore, setThresholdScore] = React.useState(0.0)
    // const [labelAnnotations, setLabelAnnotations] = React.useState([])
    // const anomalyThreshold = threshold ? threshold : 0
    // const anomaly_scores = data.map(d => d.score)
    // const [anomMarkers, setAnomMarkers] = React.useState([])

    // const onPlotReady = (plot) => {
    //     plot.on('legend-item:click', ({ view }) => {
    //         console.log(view.geometries[0].data);
    //         let shownColumn = new Set(view.geometries[0].data.map(item => item.column))
    //         setCheckedList(Array.from(shownColumn))
    //         console.log(Array.from(shownColumn))
    //     });
    // };
    let labelAnnotations = []
    for (const d of data) {
        if (d.label === 1) {
            labelAnnotations.push({
                type: 'line',
                start: [d.date, 'min'],
                end: [d.date, 'max'],
                style: {
                    stroke: 'red',
                    strokeOpacity: 0.05
                }
            })
        }
    }

    const minDate = moment(Math.min.apply(Math, data.map(o => new Date(o.date)))).format('YYYY-MM-DD HH:mm')
    const maxDate = moment(Math.max.apply(Math, data.map(o => new Date(o.date)))).format('YYYY-MM-DD HH:mm')
    console.log(minDate)

    React.useEffect(() => {
        setData(chartData)
        setThresholdScore(threshold)
    }, [chartData, threshold]);

    const mainConfig = {
        data,
        xField: 'date',
        yField: 'value',
        seriesField: 'column',
        xAxis: {
            tickCount: 24
        },
        yAxis: {
            label: {
                // 数值格式化为千分位
                formatter: (v) => `${v}`.replace(/\d{1,3}(?=(\d{3})+$)/g, (s) => `${s},`),
            },
        },
        legend: {
            position: 'bottom',
        },
        smooth: true,
        annotations: labelAnnotations,
    };

    const subConfig = {
        data,
        autoFit: false,
        height: 100,
        xField: 'date',
        yField: 'score',
        xAxis: {
            type: 'time'
        },
        color: 'rgba(255, 0, 0, 0.05)',
        columnWidthRatio: 1,
        annotations: [
            {
                type: 'line',
                start: [minDate, thresholdScore],
                end: [maxDate, thresholdScore],
                top: true,
                text: {
                    content: 'Anomaly Threshold',
                    position: '0%',
                    style: {
                        textAlign: 'left',
                        fill: 'red',
                        fontWeight: 700
                    },
                },
                style: {
                    stroke: 'red',
                    lineWidth: 2,
                    lineDash: [4, 4],
                },
            }
        ],
    }

    return <>
        <Line {...mainConfig} />
        <Column {...subConfig} />
    </>
}